from argparse import Namespace
from pathlib import Path

import pytest

from verl.implementations.launch import _adapter_command, _verl_command
from verl.implementations.presets import get_paper_reported_settings, get_preset


def _args(tmp_path, model):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    train = tmp_path / "train.parquet"
    train.touch()
    reward = tmp_path / "reward.pt"
    reward.touch()
    return Namespace(
        model=model,
        model_path=str(checkpoint),
        train_data=str(train),
        val_data=None,
        precomputed_reward=str(reward),
        output_dir=str(tmp_path / "out"),
        resume_from=None,
        training_steps=10,
        training_epochs=None,
        group_size=None,
        max_response_length=None,
        learning_rate=None,
        max_prompt_length=None,
        train_batch_size=None,
        mini_batch_size=None,
        dtype=None,
        instructblip_protocol="original",
        micro_batch_size_per_gpu=1,
        rollout_gpu_memory_utilization=0.6,
        n_gpus_per_node=8,
        save_freq=20,
        llama_model=None,
        vit_pretrained_path="",
        qformer_model="",
        ckpt="",
        bert_model_path="",
    )


def _command(tmp_path, model):
    return _verl_command(_args(tmp_path, model), get_preset(model), str(tmp_path))


def test_qwenvl_launcher_restores_legacy_remote_code_and_vllm_contract(tmp_path):
    command = _command(tmp_path, "qwenvl")
    assert "data.trust_remote_code=True" in command
    assert "actor_rollout_ref.model.trust_remote_code=True" in command
    assert "actor_rollout_ref.model.use_remove_padding=False" in command
    assert "actor_rollout_ref.model.use_fused_kernels=False" in command
    assert "actor_rollout_ref.rollout.enforce_eager=True" in command
    assert "actor_rollout_ref.rollout.enable_chunked_prefill=True" in command
    assert "actor_rollout_ref.rollout.free_cache_engine=False" in command
    assert any("QwenVLForConditionalGeneration" in item for item in command)
    assert any("c_attn|c_proj|w1|w2" in item for item in command)


def test_llava_launcher_uses_verified_multimodal_lora_surface(tmp_path):
    command = _command(tmp_path, "llava15")
    assert "actor_rollout_ref.model.use_remove_padding=False" in command
    assert "actor_rollout_ref.model.use_fused_kernels=False" in command
    assert "actor_rollout_ref.rollout.enforce_eager=True" in command
    assert "actor_rollout_ref.rollout.enable_chunked_prefill=True" in command
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0" in command
    assert 'trainer.logger=["console","wandb"]' in command
    assert "actor_rollout_ref.rollout.agent.num_workers=8" in command
    assert not any("disable_mm_preprocessor_cache" in item for item in command)
    assert "actor_rollout_ref.rollout.free_cache_engine=True" in command
    assert "algorithm.copo.fixed_trajectory=True" in command
    assert "algorithm.copo.fixed_model_family=llava15" in command
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=copo_eq11" in command
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" in command
    assert "+actor_rollout_ref.actor.use_rollout_log_probs=True" in command
    assert "trainer.val_before_train=False" in command
    assert "trainer.test_freq=-1" in command
    assert any("language_model" in item and "q_proj" in item for item in command)
    assert not any(item == "data.trust_remote_code=True" for item in command)


@pytest.mark.parametrize("model", ["llava15", "qwenvl"])
def test_native_launcher_micro_batch_divides_normalized_mini_batch(tmp_path, model):
    command = _command(tmp_path, model)
    preset = get_preset(model)
    micro = int(next(x.split("=", 1)[1] for x in command if x.startswith(
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="
    )))
    normalized_mini = preset["mini_batch_size"] * preset["group_size"] // 8
    assert normalized_mini % micro == 0


def test_qwenvl_launcher_uses_native_fixed_trajectory_eq11(tmp_path):
    command = _command(tmp_path, "qwenvl")
    assert "algorithm.copo.fixed_trajectory=True" in command
    assert "algorithm.copo.fixed_model_family=qwenvl" in command
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=copo_eq11" in command
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" in command
    assert "+actor_rollout_ref.actor.use_rollout_log_probs=True" in command
    assert "trainer.val_before_train=False" in command
    assert "trainer.test_freq=-1" in command


def test_native_training_epochs_can_increase_resume_target(tmp_path):
    args = _args(tmp_path, "llava15")
    args.training_epochs = 7
    command = _verl_command(args, get_preset("llava15"), str(tmp_path))
    assert "trainer.total_epochs=7" in command


def test_native_checkpoint_frequency_is_configurable(tmp_path):
    args = _args(tmp_path, "llava15")
    args.save_freq = 1
    command = _verl_command(args, get_preset("llava15"), str(tmp_path))
    assert "trainer.save_freq=1" in command


def test_native_resource_overrides_do_not_change_presets(tmp_path):
    args = _args(tmp_path, "llava15")
    args.group_size = 2
    args.max_response_length = 128
    args.micro_batch_size_per_gpu = 2
    args.rollout_gpu_memory_utilization = 0.45
    args.max_prompt_length = 96
    args.train_batch_size = 4
    args.mini_batch_size = 2
    command = _verl_command(args, get_preset("llava15"), str(tmp_path))
    assert "actor_rollout_ref.rollout.n=2" in command
    assert "actor_rollout_ref.rollout.agent.num_workers=2" in command
    assert "data.max_response_length=128" in command
    assert "data.max_prompt_length=96" in command
    assert "data.train_batch_size=4" in command
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=2" in command
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2" in command
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2" in command
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2" in command
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.45" in command
    assert get_preset("llava15")["group_size"] == 8
    assert get_preset("llava15")["max_response_length"] == 8192


def test_learning_rate_override_is_shared_without_mutating_presets(tmp_path):
    native_root = tmp_path / "native"
    native_root.mkdir()
    native_args = _args(native_root, "llava15")
    native_args.learning_rate = 2e-6
    native = _verl_command(native_args, get_preset("llava15"), str(native_root))
    assert "actor_rollout_ref.actor.optim.lr=2e-06" in native

    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    adapter_args = _args(adapter_root, "instructblip")
    adapter_args.learning_rate = 1e-6
    adapter = _adapter_command(adapter_args, get_preset("instructblip"), str(adapter_root))
    assert adapter[adapter.index("--lr") + 1] == "1e-06"
    assert get_preset("instructblip")["lr"] == 1e-5


def test_appendix_g_settings_are_separate_reference_metadata():
    paper = get_paper_reported_settings()
    assert paper["train_batch_size"] == 512
    assert paper["mini_batch_size"] == 256
    assert paper["max_prompt_length"] == 1024
    assert paper["max_response_length"] == 8192
    assert paper["lr"] == 1e-6
    assert paper["vllm_gpu_memory_utilization"] == 0.6
    paper["lr"] = 99
    assert get_paper_reported_settings()["lr"] == 1e-6


def test_native_launcher_rejects_checkpoint_context_overflow(tmp_path):
    args = _args(tmp_path, "llava15")
    (Path(args.model_path) / "config.json").write_text(
        '{"text_config":{"max_position_embeddings":4096}}'
    )
    with pytest.raises(SystemExit, match="checkpoint context limit"):
        _verl_command(args, get_preset("llava15"), str(tmp_path))
    args.max_prompt_length = 512
    args.max_response_length = 512
    command = _verl_command(args, get_preset("llava15"), str(tmp_path))
    assert "data.max_prompt_length=512" in command
    assert "data.max_response_length=512" in command


def test_instructblip_launcher_requires_fixed_v3_eq11_parameters(tmp_path):
    args = _args(tmp_path, "instructblip")
    command = _adapter_command(args, get_preset("instructblip"), str(tmp_path))
    assert "--fixed-trajectory" in command
    assert command[command.index("--clip-ratio") + 1] == "0.2"
    assert command[command.index("--kl-coef") + 1] == "0.01"
    assert "dummy" not in command
    assert command[command.index("--instructblip-protocol") + 1] == "original"


def test_instructblip_optional_protocol_is_explicitly_forwarded(tmp_path):
    args = _args(tmp_path, "instructblip")
    args.instructblip_protocol = "split_context_two_shot"
    command = _adapter_command(args, get_preset("instructblip"), str(tmp_path))
    assert command[command.index("--instructblip-protocol") + 1] == (
        "split_context_two_shot"
    )


def test_adapter_resource_and_dtype_overrides_are_forwarded(tmp_path):
    args = _args(tmp_path, "instructblip")
    args.group_size = 2
    args.max_response_length = 128
    args.dtype = "bfloat16"
    command = _adapter_command(args, get_preset("instructblip"), str(tmp_path))
    assert command[command.index("--group-size") + 1] == "2"
    assert command[command.index("--max-new-tokens") + 1] == "128"
    assert command[command.index("--dtype") + 1] == "bfloat16"


def test_minigpt4_launcher_requires_fixed_v3_eq11_parameters(tmp_path):
    args = _args(tmp_path, "minigpt4")
    for name in ("vit", "qformer", "stage2", "bert"):
        (tmp_path / name).touch()
    args.vit_pretrained_path = str(tmp_path / "vit")
    args.qformer_model = str(tmp_path / "qformer")
    args.ckpt = str(tmp_path / "stage2")
    args.bert_model_path = str(tmp_path / "bert")
    command = _adapter_command(args, get_preset("minigpt4"), str(tmp_path))
    assert "--fixed-trajectory" in command
    assert command[command.index("--clip-ratio") + 1] == "0.2"
    assert command[command.index("--kl-coef") + 1] == "0.01"
    assert "dummy" not in command


def test_adapter_launcher_rejects_historical_smoke_step_default(tmp_path):
    args = _args(tmp_path, "instructblip")
    args.training_steps = None
    with pytest.raises(SystemExit, match="positive explicit value"):
        _adapter_command(args, get_preset("instructblip"), str(tmp_path))


@pytest.mark.parametrize("model", ["instructblip", "minigpt4"])
def test_adapter_launcher_rejects_missing_resume_state_before_training(tmp_path, model):
    args = _args(tmp_path, model)
    args.resume_from = str(tmp_path / "missing-state.pt")
    with pytest.raises(SystemExit, match="--resume-from path does not exist"):
        _adapter_command(args, get_preset(model), str(tmp_path))
