# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.causal_reward.trajectory import multimodal_content_hash
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _load_fixed_legacy_images(messages: list[dict]) -> list[Image.Image]:
    """Load the unresized local images used by the legacy fixed collectors."""
    images = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for segment in content:
            if not isinstance(segment, dict) or segment.get("type") != "image":
                continue
            value = segment.get("image")
            if isinstance(value, Image.Image):
                image = value
            elif value is None and "bytes" in segment:
                image = Image.open(BytesIO(segment["bytes"]))
            else:
                parsed = urlparse(str(value))
                if parsed.scheme not in ("", "file"):
                    raise ValueError("fixed LLaVA/Qwen-VL trajectories require local image inputs")
                path = Path(unquote(parsed.path if parsed.scheme == "file" else str(value)))
                if not path.is_file():
                    raise FileNotFoundError(f"fixed trajectory image does not exist: {path}")
                image = Image.open(path)
            images.append(image.convert("RGB"))
    return images


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        fixed_response_ids = kwargs.get("copo_fixed_response_token_ids")
        fixed_family = getattr(getattr(getattr(self.config, "algorithm", None), "copo", None),
                               "fixed_model_family", None) if hasattr(self, "config") else None
        if fixed_response_ids is not None and fixed_family in {"llava15", "qwenvl"}:
            multi_modal_data = {"images": _load_fixed_legacy_images(messages)}
        else:
            multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )
        prefix_ids = kwargs.get("copo_prefix_ids")
        if prefix_ids is not None:
            prompt_ids = prompt_ids + [int(token) for token in prefix_ids]

        if fixed_response_ids is not None:
            if prefix_ids is not None:
                raise ValueError("fixed trajectories cannot be combined with a counterfactual prompt prefix")
            fixed_prompt_suffix = kwargs.get("copo_fixed_prompt_suffix_token_ids", ())
            prompt_ids = prompt_ids + [int(token) for token in fixed_prompt_suffix]
            response_ids = [int(token) for token in fixed_response_ids]
            if not response_ids:
                raise ValueError("fixed trajectory response must contain at least one generated token")
            if len(response_ids) > self.response_length:
                raise ValueError(
                    f"fixed trajectory response length {len(response_ids)} exceeds configured limit "
                    f"{self.response_length}"
                )
            if "copo_fixed_outcome_reward" not in kwargs:
                raise ValueError("fixed trajectory is missing its outcome reward")
            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=None,
                multi_modal_data=multi_modal_data,
                reward_score=float(kwargs["copo_fixed_outcome_reward"]),
                num_turns=2,
                metrics=AgentLoopMetrics(),
            )
            output.extra_fields["generated_text"] = self.tokenizer.decode(
                output.response_ids, skip_special_tokens=True
            )
            output.extra_fields["copo_actual_image_hash"] = multimodal_content_hash(
                {"images": images, "videos": videos}
            )
            output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
            return output

        copo_max_new_tokens = kwargs.get("copo_max_new_tokens")
        if copo_max_new_tokens is not None:
            sampling_params = dict(sampling_params)
            sampling_params["max_tokens"] = int(copo_max_new_tokens)

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )

        output.extra_fields["generated_text"] = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
