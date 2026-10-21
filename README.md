# [COPO: Causal-Oriented Policy Optimization for Hallucinations of MLLMs](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_COPO_Causal-Oriented_Policy_Optimization_for_Hallucinations_of_MLLMs_CVPR_2026_paper.html)

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-gold?style=flat-square)](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_COPO_Causal-Oriented_Policy_Optimization_for_Hallucinations_of_MLLMs_CVPR_2026_paper.html)
![MLLM](https://img.shields.io/badge/MLLM-purple?style=flat-square)
[![verl](https://img.shields.io/badge/verl-orange?style=flat-square)](https://github.com/volcengine/verl)

This repository contains the PyTorch implementation of our CVPR 2026 paper, "COPO:
Causal-Oriented Policy Optimization for Hallucinations of MLLMs."

## Introduction

Multimodal large language models can produce reasoning and answers that are not adequately
supported by visual evidence. Policy optimization based only on final outcomes offers
limited insight into how individual reasoning tokens affect a response. COPO addresses
this limitation by estimating token-level causal contributions through Causal Sufficiency
and Causal Necessity, and incorporating this feedback into policy optimization to mitigate
multimodal hallucinations. The implementation is built upon
[verl](https://github.com/volcengine/verl).

## Installation

`environment.yml` defines the Python 3.10 environment and installs both the project
dependencies and the vendored `verl` package. Create the environment once, then
activate it whenever you work with COPO:

```bash
conda env create -f environment.yml
conda activate copo
```

After activation, the following script runs the repository's basic import and source
checks. It is a useful installation check, but it does not load a model or start training.

```bash
bash scripts/check.sh
```

## Quick Start

The following example runs the complete workflow with LLaVA-1.5. Supply compatible base
and reference checkpoints, task images, and real annotations; model weights are not
included in this repository.

### 1. Prepare Data

Create a JSON Lines source file containing an image, a question, and its ground-truth
task answer. Each row must include the fields shown below:

```json
{"sample_id":"sample-0001","group_id":"group-0001","image_id":"image-0001","image":"path/to/image.jpg","instruction":"Is there a dog in the image?","answer":"yes","task":"vqa"}
```

`sample_id` and `group_id` identify the example and its rollout group.
`image_id` identifies the image, while `image` is the path to the local image file.
`instruction`, `answer`, and `task` contain the question, its real task label, and
the task type. Replace all example values and paths with your own data.

Validate the source data and build the files used by collection and training:

```bash
python scripts/prepare_data/build_training_data.py \
    --input path/to/source.jsonl \
    --canonical-output path/to/canonical.jsonl \
    --native-output path/to/native.json
```

`--input` points to the source JSON Lines file. The command generates two files:
`canonical.jsonl`, which is used in the next step to collect trajectories, and
`native.json`, which provides the training data for LLaVA-1.5. Both output paths
can be changed as needed.

### 2. Generate Causal Rewards

Specify the policy used to collect trajectories, a compatible reference model, the
canonical dataset, and the destination for the collected output:

```bash
python scripts/prepare_reward/generate_rewards.py \
    --model llava15 \
    --model-path path/to/llava-1.5 \
    --reference-model path/to/reference-model \
    --dataset path/to/canonical.jsonl \
    --max-response-length 1024 \
    --output path/to/trajectories.pt
```

The resulting `trajectories.pt` contains the trajectories and causal rewards consumed
by the training step. In this command:

- `--model` selects the model family used for collection.
- `--model-path` points to the local policy checkpoint.
- `--reference-model` points to the compatible reference checkpoint.
- `--dataset` reads the `canonical.jsonl` file created in Step 1.
- `--output` chooses where the collected file is saved.

### 3. Train

Train the LLaVA-1.5 policy using the native training data and the trajectory file created
above. `scripts/train.sh` is the public training entry point:

```bash
bash scripts/train.sh \
    --model llava15 \
    --model-path path/to/llava-1.5 \
    --train-data path/to/native.json \
    --precomputed-reward path/to/trajectories.pt \
    --max-prompt-length 1024 \
    --max-response-length 1024 \
    --output-dir path/to/training-output
```

`--model-path` selects the policy to update, `--train-data` supplies its training
samples, `--precomputed-reward` loads the previous step's output, and `--output-dir`
sets the checkpoint directory. Keep collection and training settings such as response
length consistent, and keep the total prompt and response length within the checkpoint's
context window. Replace every example path before starting the run; training results are
written below the directory supplied through `--output-dir`.

## Full Pipeline

### Files and Commands

The main inputs, outputs, and entry points are:

| File or command | Purpose |
|---|---|
| `environment.yml` | Conda environment configuration |
| `scripts/check.sh` | Basic import and source checks |
| `source.jsonl` | User-provided images, questions, and task labels |
| `scripts/prepare_data/build_training_data.py` | Validates source data and prepares both training formats |
| `canonical.jsonl` | Input to trajectory collection and InstructBLIP/MiniGPT-4 training |
| `native.json` | Training input for LLaVA-1.5 and original Qwen-VL |
| `scripts/prepare_reward/generate_rewards.py` | Generates trajectories and causal rewards |
| `trajectories.pt` | Collected output passed to training |
| `scripts/train.sh` | Training entry point |
| `--output-dir` | Destination for training outputs and checkpoints |

### Choose a Model

Use the same data, collection, and training entry points with one of these `--model`
values:

| `--model` | Model | Training data |
|---|---|---|
| `llava15` | LLaVA-1.5 | `native.json` |
| `qwenvl` | Original Qwen-VL | `native.json` |
| `instructblip` | InstructBLIP | `canonical.jsonl` |
| `minigpt4` | MiniGPT-4 | `canonical.jsonl` |

Qwen-VL here refers to the original Qwen-VL family, not Qwen2.5-VL. Use the same model
selector for collection and training, with checkpoints and processors that belong to
that model family.

For original Qwen-VL, replace `llava15` with `qwenvl` in both Quick Start commands
and provide Qwen-VL-compatible policy and reference checkpoints. Its training input
remains `native.json`.

InstructBLIP and MiniGPT-4 use `canonical.jsonl` for training and require an explicit
`--training-steps` value. MiniGPT-4 also requires matching language-model, vision,
Q-Former, BERT, and stage-2 weights. View all model-specific arguments with:

```bash
python scripts/prepare_reward/generate_rewards.py --help
bash scripts/train.sh --help
```

### Configure, Save, and Resume Training

Keep `--model`, the response-length limit, and any rollout-group override consistent
between collection and training. Use `--max-prompt-length` and
`--max-response-length` values that fit the selected checkpoint, and adjust training
options through `scripts/train.sh`.

Common overrides include `--learning-rate`, `--group-size`,
`--train-batch-size`, `--mini-batch-size`, and `--n-gpus-per-node`. If
`--group-size` is changed for training, use the same value as `--num-rollouts`
during collection. Check the available defaults and options before launching a run:

```bash
bash scripts/train.sh --help
```

`--output-dir` controls where results are saved. For LLaVA-1.5 and Qwen-VL,
`--save-freq` controls checkpoint frequency. Rerunning the same model with the same
output directory resumes from the latest checkpoint; increase `--training-epochs` to
extend the run.

For InstructBLIP and MiniGPT-4, pass their saved training-state file with
`--resume-from`. Set `--training-steps` to the new total number of steps, which must
be greater than the step recorded in that file.

### Collect the Next Trajectory Round

When continuing from an updated policy, collect a new trajectory file before the next
training round. A native LLaVA-1.5 or Qwen-VL checkpoint writes a collector-ready
adapter under `<output-dir>/global_step_N/actor/lora_adapter`, where `N` is the saved
training step. For example, use a saved Qwen-VL adapter as the collector state:

```bash
python scripts/prepare_reward/generate_rewards.py \
    --model qwenvl \
    --model-path path/to/qwen-vl \
    --reference-model path/to/reference-model \
    --dataset path/to/canonical.jsonl \
    --collector-state path/to/qwenvl-output/global_step_N/actor/lora_adapter \
    --collection-round 1 \
    --max-response-length 1024 \
    --output path/to/qwenvl-trajectories-round-1.pt
```

To continue Qwen-VL training, rerun `bash scripts/train.sh` with `--model qwenvl`,
keeping the same `--model-path`, `--train-data`, `--output-dir`, and length/group
settings. Set `--precomputed-reward` to
`path/to/qwenvl-trajectories-round-1.pt`. If the previous run has reached its
epoch target, increase `--training-epochs` to the new total epoch target. For InstructBLIP and MiniGPT-4,
`--collector-state` accepts their saved training-state file instead. Trajectory
collection and training remain separate steps; repeat them only when a refreshed round
is needed. Replace `N` and all other example paths with the files produced by your run.

## Citation

```bibtex
@inproceedings{guo2026copo,
  title={{COPO}: Causal-Oriented Policy Optimization for Hallucinations of {MLLMs}},
  author={Guo, Peizheng and Wang, Jingyao and Qiang, Wenwen and Zhou, Jiahuan and Zheng, Changwen and Hua, Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11051--11063},
  year={2026}
}
```
