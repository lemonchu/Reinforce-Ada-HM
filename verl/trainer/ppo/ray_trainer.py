# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import math
import random
import re
import json
import os
import uuid
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

# Utility functions for Reinforce-Ada
from verl.trainer.ppo.reinforce_ada_utils import (
    get_first_dim_size,
    concat_dataproto_fragments,
    build_uid_to_fields_mapping,
    ensure_uid_in_batch,
    align_context_to_selected,
    merge_context_fields_into_batch,
    validate_tensordict_performance,
    compute_seq_rewards_for_round,
)

DEFAULT_REFINE_INSTRUCTION = (
    "Follow this instruction, carefully review your previous solution:\n"
    "1. Go through each calculation step-by-step. Check if there are any errors in calculations, logic, or problem understanding.\n"
    "2. If you find any mistakes, explicitly point out what was wrong and explain the correct approach.\n"
    "3. If the solution is already correct, verify each step and explain it more clearly.\n"
    "4. Finally, after finishing the review, provide your refined solution and answer.\n"
)

DEFAULT_REWRITE_INSTRUCTION = (
    "Based on your previous solution and review, solve this problem again from scratch and write a complete, "
    "self-contained solution. Your previous answer and your review will not be kept, so your final response must "
    "stand alone and include all necessary reasoning. Keep the mathematical meaning and final answer unchanged, "
    "but use a different presentation or reasoning path where possible. Write as if you are solving the problem "
    "for the first time: do not mention the previous solution, the review, re-examining, verification, or that "
    "the solution has no errors. Avoid meta-style openings or closings such as \"After re-examining the problem "
    "and the calculations,\" or \"This method clearly shows all necessary steps and ensures that there are no "
    "errors in the solution process.\" Put the final answer inside \\boxed{}."
)


def _rewrite_normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|\\[A-Za-z]+|[+\-*/=^(){}\[\].,]", text.lower())


def _rewrite_rouge_n_f1(reference: str, candidate: str, n: int = 1) -> float:
    ref_tokens = _rewrite_normalize_words(reference)
    cand_tokens = _rewrite_normalize_words(candidate)
    if len(ref_tokens) < n or len(cand_tokens) < n:
        return 0.0
    ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    cand_ngrams = Counter(tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1))
    overlap = sum((ref_ngrams & cand_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(cand_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _rewrite_lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rewrite_rouge_l_f1(reference: str, candidate: str) -> float:
    ref_tokens = _rewrite_normalize_words(reference)
    cand_tokens = _rewrite_normalize_words(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _rewrite_lcs_len(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rewrite_soft_target_score(value: float, target: float, width: float) -> float:
    width = max(width, 1e-6)
    return math.exp(-((value - target) / width) ** 2)


def _rewrite_length_score_piecewise(
    rewrite_len: int,
    reference_len: int,
    refine_len: int,
    width: float,
    low_scale: float,
    high_scale: float,
) -> tuple[float, float, int, int]:
    low_scale = max(low_scale, 1e-6)
    high_scale = max(high_scale, low_scale)
    low = max(1, round(min(reference_len, refine_len) * low_scale))
    high = max(low, round(max(reference_len, refine_len) * high_scale))
    if low <= rewrite_len <= high:
        return 1.0, 1.0, low, high
    len_ratio_to_band = rewrite_len / low if rewrite_len < low else high / max(rewrite_len, 1)
    width = max(width, 1e-6)
    score = math.exp(-(math.log(max(len_ratio_to_band, 1e-6)) / width) ** 2)
    return score, len_ratio_to_band, low, high


def _rewrite_composite_reward(
    correct: bool,
    reference: str,
    refined: str,
    rewrite: str,
    rouge_target: float,
    rouge_width: float,
    length_width: float,
    length_low_scale: float,
    length_high_scale: float,
) -> dict[str, float | int]:
    rouge1 = _rewrite_rouge_n_f1(reference, rewrite, n=1)
    rougel = _rewrite_rouge_l_f1(reference, rewrite)
    rouge = 0.5 * rouge1 + 0.5 * rougel
    rouge_score = _rewrite_soft_target_score(rouge, target=rouge_target, width=rouge_width)
    reference_len = len(_rewrite_normalize_words(reference))
    refine_len = len(_rewrite_normalize_words(refined))
    rewrite_len = len(_rewrite_normalize_words(rewrite))
    length_score, len_ratio_to_band, length_low, length_high = _rewrite_length_score_piecewise(
        rewrite_len=rewrite_len,
        reference_len=reference_len,
        refine_len=refine_len,
        width=length_width,
        low_scale=length_low_scale,
        high_scale=length_high_scale,
    )
    reward = float(correct) * rouge_score * length_score
    return {
        "reward": float(reward),
        "rouge1_f": float(rouge1),
        "rouge_l_f": float(rougel),
        "rouge_mix": float(rouge),
        "rouge_score": float(rouge_score),
        "reference_len": reference_len,
        "refine_len": refine_len,
        "rewrite_len": rewrite_len,
        "length_low": length_low,
        "length_high": length_high,
        "len_ratio_to_band": float(len_ratio_to_band),
        "length_score": float(length_score),
    }


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # For GRPO with global stats estimation, get the global pos/neg counts
        grpo_uid_to_pos_count = None
        grpo_uid_to_neg_count = None
        if hasattr(data, "meta_info") and data.meta_info is not None:
            grpo_uid_to_pos_count = data.meta_info.get("grpo_uid_to_pos_count", None)
            grpo_uid_to_neg_count = data.meta_info.get("grpo_uid_to_neg_count", None)

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
            grpo_uid_to_pos_count=grpo_uid_to_pos_count,
            grpo_uid_to_neg_count=grpo_uid_to_neg_count,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _generate_multi_round_adaptive_downsampling(
        self,
        orig_prompt_batch: DataProto,
        positive_threshold: float = 0.7,
        max_rounds: int = 4,
        round_repeat: int = 8,
        final_keep_per_prompt: int = 4,
        timing_raw: dict | None = None,
        context_batch: DataProto | None = None,
    ):
        """
        Iterative multi-round generation with early stopping downsampling.

        Args:
            orig_prompt_batch: Original prompt batch to generate from
            positive_threshold: Threshold for classifying samples as positive (reward > threshold)
            max_rounds: Maximum number of rounds to perform
            round_repeat: Number of samples to generate per active prompt in each round
            final_keep_per_prompt: Final number of samples to keep per prompt (target: half positive, half negative)
            timing_raw: Optional dict to record timing information
            context_batch: Optional context batch for field alignment via uid

        Returns:
            Tuple of (final_batch, rounds_info) where:
                - final_batch: DataProto with selected samples and aligned context fields
                - rounds_info: Dict with per-round statistics
        """
        # Build uid -> fields mapping from context
        ctx_uid_to_fields = {}
        if context_batch is not None:
            ctx_uid_to_fields = build_uid_to_fields_mapping(context_batch)

        # Ensure orig_prompt_batch has uid
        ensure_uid_in_batch(orig_prompt_batch, context_batch)
        uid_arr = list(orig_prompt_batch.non_tensor_batch["uid"])

        # Initialize state tracking for each uid
        state = {uid: {"finished": False, "seen": 0, "pos": 0, "neg": 0} for uid in uid_arr}

        # Caches for positive and negative samples per uid
        pos_cache = defaultdict(list)
        neg_cache = defaultdict(list)
        selected_pool_batches: list[DataProto] = []
        selected_count_by_uid = defaultdict(int)
        # For GRPO with global statistics estimation
        uid_full_stats = {uid: {"total_pos": 0, "total_neg": 0} for uid in uid_arr}
        rounds_info = {"per_round": []}

        # Main generation loop
        active_uids = set(uid_arr)
        for r in range(max_rounds):
            t0 = time.time()
            if not active_uids:
                rounds_info["per_round"].append(
                    {
                        "round": r,
                        "active_prompts": 0,
                        "completed": 0,
                        "finished_prompts": sum(1 for s in state.values() if s["finished"]),
                        "sec": 0.0,
                    }
                )
                break

            # Create mini-batch for active prompts only
            uid_to_idx = {uid: i for i, uid in enumerate(uid_arr)}
            active_indices = [uid_to_idx[uid] for uid in uid_arr if uid in active_uids]
            mini_prompt_batch = orig_prompt_batch[active_indices]
            round_inp = mini_prompt_batch.repeat(repeat_times=round_repeat, interleave=True)

            # Pad to be divisible by dp_size
            dp_size = self.actor_rollout_wg.dp_size if hasattr(self.actor_rollout_wg, "dp_size") else 8
            batch_size = len(round_inp)
            padding_applied = False
            if batch_size % dp_size != 0:
                padding_needed = dp_size - (batch_size % dp_size)
                print(
                    f"Padding batch from {batch_size} to {batch_size + padding_needed} "
                    f"to make it divisible by {dp_size}"
                )
                indices_to_repeat = list(range(batch_size - padding_needed, batch_size))
                if len(indices_to_repeat) == 0:
                    indices_to_repeat = [batch_size - 1] * padding_needed
                padding_batch = round_inp[indices_to_repeat]
                round_inp = DataProto.concat([round_inp, padding_batch])
                padding_applied = True

            # Generate sequences
            gen_out = (
                self.actor_rollout_wg.generate_sequences(round_inp)
                if not self.async_rollout_mode
                else self.async_rollout_manager.generate_sequences(round_inp)
            )

            # Remove padding if applied
            if padding_applied:
                gen_out = gen_out[:batch_size]
                round_inp = round_inp[:batch_size]

            # Compute rewards for this round
            mini_with_out, seq_reward, uids_round = compute_seq_rewards_for_round(
                mini_prompt_batch=mini_prompt_batch,
                gen_out=gen_out,
                ctx_uid_to_fields=ctx_uid_to_fields,
                reward_fn=self.reward_fn,
                use_rm=self.use_rm,
                rm_wg=self.rm_wg,
                config=self.config,
                kl_ctrl_in_reward=self.kl_ctrl_in_reward if self.config.algorithm.use_kl_in_reward else None,
            )
            seq_reward_np = seq_reward.detach().cpu().numpy().tolist()

            # Group by uid
            per_uid_local_idx = defaultdict(list)
            for j, uid in enumerate(uids_round):
                per_uid_local_idx[uid].append(j)

            # Update state and cache samples
            completed_this_round = 0
            for uid in list(active_uids):
                locs = per_uid_local_idx.get(uid, [])
                if not locs:
                    continue
                st = state[uid]

                # Cache positive and negative samples
                for j in locs:
                    if st["finished"]:
                        break
                    st["seen"] += 1
                    is_positive = seq_reward_np[j] > positive_threshold
                    if is_positive:
                        st["pos"] += 1
                        pos_cache[uid].append(mini_with_out[[j]])
                        uid_full_stats[uid]["total_pos"] += 1
                    else:
                        st["neg"] += 1
                        neg_cache[uid].append(mini_with_out[[j]])
                        uid_full_stats[uid]["total_neg"] += 1

                def downsample_cache(pos_cache, neg_cache, uid, target_total):
                    target_pos = min(target_total // 2, len(pos_cache[uid]))
                    target_neg = min(target_total - target_pos, len(neg_cache[uid]))
                    if target_pos + target_neg < target_total:
                        # Not enough total samples, adjust targets
                        if len(pos_cache[uid]) > target_pos:
                            additional_pos = min(len(pos_cache[uid]) - target_pos, target_total - (target_pos + target_neg))
                            target_pos += additional_pos
                        elif len(neg_cache[uid]) > target_neg:
                            additional_neg = min(len(neg_cache[uid]) - target_neg, target_total - (target_pos + target_neg))
                            target_neg += additional_neg
                    pos_frags = pos_cache[uid][:target_pos]
                    neg_frags = neg_cache[uid][:target_neg]
                    merged = concat_dataproto_fragments(pos_frags + neg_frags)
                    return merged

                # Check if we have enough samples to finish this uid
                if not st["finished"]:
                    if self.config.algorithm.reinforce_ada_choice == "balanced":
                        target_pos = final_keep_per_prompt // 2
                        target_neg = final_keep_per_prompt - target_pos

                        if len(pos_cache[uid]) >= target_pos and len(neg_cache[uid]) >= target_neg:
                            merged = downsample_cache(pos_cache, neg_cache, uid, final_keep_per_prompt)
                            selected_pool_batches.append(merged)
                            selected_count_by_uid[uid] = get_first_dim_size(merged)
                            st["finished"] = True
                            completed_this_round += 1

                    else:  # positive_focused
                        assert self.config.algorithm.reinforce_ada_choice == "positive_focused", (
                            "reinforce_ada_choice has to be one of {'balanced', 'positive_focused'}"
                        )
                        target_pos = 1
                        if len(pos_cache[uid]) >= target_pos:
                            merged = downsample_cache(pos_cache, neg_cache, uid, final_keep_per_prompt)
                            selected_pool_batches.append(merged)
                            selected_count_by_uid[uid] = get_first_dim_size(merged)
                            st["finished"] = True
                            completed_this_round += 1

            # Update active set
            active_uids = {u for u in active_uids if not state[u]["finished"]}

            # Record timing and stats
            sec = time.time() - t0
            if timing_raw is not None:
                timing_raw[f"gen_round_{r}_sec"] = sec

            rounds_info["per_round"].append(
                {
                    "round": r,
                    "active_prompts": len(per_uid_local_idx),
                    "completed": completed_this_round,
                    "finished_prompts": sum(1 for s in state.values() if s["finished"]),
                    "reward_mean": float(np.mean(seq_reward_np)) if seq_reward_np else 0.0,
                    "sec": round(sec, 3),
                }
            )
            print(
                f"[Gen-Round {r}] active_prompts={len(per_uid_local_idx)} "
                f"completed={completed_this_round} "
                f"finished={rounds_info['per_round'][-1]['finished_prompts']} "
                f"time={sec:.3f}s "
                f"reward_mean={rounds_info['per_round'][-1]['reward_mean']:.4f}"
            )

            if not active_uids:
                break

        # Handle fallback for uids that didn't reach target
        uids_that_need_fallback = {uid for uid in uid_arr if not state[uid]["finished"]}

        for uid in uids_that_need_fallback:
            if uid in pos_cache or uid in neg_cache:
                pos_num = len(pos_cache[uid])
                neg_num = len(neg_cache[uid])
                n_rows = pos_num + neg_num
                take = min(final_keep_per_prompt, n_rows)

                if n_rows < final_keep_per_prompt:
                    print(
                        f"[WARN] uid={uid} has {n_rows} samples, less than target "
                        f"{final_keep_per_prompt}, but continuing"
                    )

                if self.config.algorithm.reinforce_ada_choice == "positive_focused":
                    ratio = (pos_num / n_rows) if n_rows > 0 else 0.0
                    target_pos = math.ceil(ratio * final_keep_per_prompt)
                    target_pos = max(min(target_pos, take - 1), 1)
                    target_neg = take - target_pos

                actual_pos = min(pos_num, target_pos)
                actual_neg = min(neg_num, target_neg)

                # If one type is insufficient, fill with the other
                if actual_pos + actual_neg < take:
                    if pos_num > actual_pos:
                        additional_pos = min(pos_num - actual_pos, take - actual_pos - actual_neg)
                        actual_pos += additional_pos
                    elif neg_num > actual_neg:
                        additional_neg = min(neg_num - actual_neg, take - actual_pos - actual_neg)
                        actual_neg += additional_neg

                keep_pos = actual_pos
                keep_neg = actual_neg
                pos_frags = pos_cache[uid][:keep_pos] if keep_pos > 0 else []
                neg_frags = neg_cache[uid][:keep_neg] if keep_neg > 0 else []
                frags_to_merge = pos_frags + neg_frags

                if frags_to_merge:
                    merged = concat_dataproto_fragments(frags_to_merge)
                    selected_pool_batches.append(merged)
                    selected_count_by_uid[uid] = get_first_dim_size(merged)
                else:
                    print(f"[WARN] uid={uid} frags_to_merge is empty, cannot fallback")
            else:
                print(f"[WARN] uid={uid} not in pos_cache or neg_cache, cannot fallback")

        if not selected_pool_batches:
            raise RuntimeError(
                "No samples selected after early stopping. Check if threshold/rules are too strict or data is abnormal"
            )

        # Concatenate all selected samples
        selected_batch = concat_dataproto_fragments(selected_pool_batches)

        # Align context fields to selected batch
        _context_src = context_batch if context_batch is not None else orig_prompt_batch
        ctx_rows = align_context_to_selected(selected_batch, _context_src)

        # Merge missing fields from context into selected batch
        merge_context_fields_into_batch(selected_batch, ctx_rows)

        final_batch = selected_batch

        # Ensure token_level_scores exists (fallback to token_level_rewards)
        if "token_level_scores" not in final_batch.batch and "token_level_rewards" in final_batch.batch:
            final_batch.batch["token_level_scores"] = final_batch.batch["token_level_rewards"]

        # For GRPO with global stats, log pos/neg counts
        uid_to_pos_count = {uid: stats["total_pos"] for uid, stats in uid_full_stats.items()}
        uid_to_neg_count = {uid: stats["total_neg"] for uid, stats in uid_full_stats.items()}

        if not hasattr(final_batch, "meta_info") or final_batch.meta_info is None:
            final_batch.meta_info = {}
        final_batch.meta_info["grpo_uid_to_pos_count"] = uid_to_pos_count
        final_batch.meta_info["grpo_uid_to_neg_count"] = uid_to_neg_count

        # Validate that we maintained efficient TensorDict structure
        validate_tensordict_performance(final_batch, context="final_batch")

        return final_batch, rounds_info

    def _extract_last_user_message(self, raw_prompt) -> str:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, str):
            return raw_prompt
        if not isinstance(raw_prompt, list):
            return ""

        for message in reversed(raw_prompt):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                return "".join(parts)
            if isinstance(content, str):
                return content
            return str(content)
        return ""

    @staticmethod
    def _extract_role_message(raw_prompt, role: str, first: bool = True) -> str:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, str):
            return raw_prompt if role == "user" else ""
        if not isinstance(raw_prompt, list):
            return ""
        messages = raw_prompt if first else list(reversed(raw_prompt))
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != role:
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                return "".join(parts)
            return content if isinstance(content, str) else str(content)
        return ""

    def _decode_response_text(self, batch: DataProto, row_idx: int) -> str:
        response_ids = batch.batch["responses"][row_idx]
        if "response_mask" in batch.batch.keys():
            response_ids = response_ids[batch.batch["response_mask"][row_idx].bool()]
        return self.tokenizer.decode(response_ids.tolist(), skip_special_tokens=True).strip()

    def _select_refine_source_batch(self, final_batch: DataProto) -> tuple[Optional[DataProto], dict[str, float]]:
        cfg = self.config.algorithm
        mode = str(cfg.get("refine_selection_mode", "difficult_first"))
        difficult_space = int(cfg.get("difficult_refine_space", 256) or 0)
        positive_threshold = float(cfg.get("positive_threshold", 0.7))
        seed = int(self.config.data.get("seed", 0) or 0) + int(self.global_steps)
        rng = random.Random(seed)

        if "uid" not in final_batch.non_tensor_batch:
            return None, {"refine/source_prompts": 0}

        reward_key = "token_level_scores" if "token_level_scores" in final_batch.batch.keys() else "token_level_rewards"
        if reward_key not in final_batch.batch.keys():
            return None, {"refine/source_prompts": 0}

        seq_scores = final_batch.batch[reward_key].sum(dim=-1).detach().cpu().numpy()
        uids = list(final_batch.non_tensor_batch["uid"])
        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid_to_indices[uid].append(idx)

        uid_order = list(uid_to_indices.keys())
        uid_mean = {
            uid: float(np.mean([seq_scores[idx] for idx in idxs])) for uid, idxs in uid_to_indices.items()
        }
        selected: list[int] = []
        difficult_selected = 0
        difficult_pos_selected = 0
        difficult_zero_selected = 0

        if mode in {"random", "uniform"}:
            for uid in uid_order:
                selected.append(rng.choice(uid_to_indices[uid]))
        elif mode in {"difficult_first", "difficult", "hard_first"}:
            sorted_uids = sorted(uid_order, key=lambda uid: (uid_mean[uid], uid))
            difficult_uids = set(sorted_uids[: max(0, min(difficult_space, len(sorted_uids)))])
            for uid in sorted_uids:
                idxs = uid_to_indices[uid]
                if uid in difficult_uids:
                    pos = [idx for idx in idxs if seq_scores[idx] > positive_threshold]
                    neg = [idx for idx in idxs if seq_scores[idx] <= positive_threshold]
                    if pos:
                        selected.append(rng.choice(pos))
                        difficult_pos_selected += 1
                    elif neg:
                        selected.append(rng.choice(neg))
                        difficult_zero_selected += 1
                    else:
                        selected.append(rng.choice(idxs))
                    difficult_selected += 1
                else:
                    selected.append(rng.choice(idxs))
        else:
            raise ValueError(f"Unknown refine_selection_mode: {mode}")

        if not selected:
            return None, {"refine/source_prompts": 0}

        source_batch = final_batch[selected]
        metrics = {
            "refine/source_prompts": len(selected),
            "refine/difficult_space": min(difficult_space, len(uid_order)) if mode not in {"random", "uniform"} else 0,
            "refine/difficult_selected": difficult_selected,
            "refine/difficult_pos_selected": difficult_pos_selected,
            "refine/difficult_zero_selected": difficult_zero_selected,
            "refine/source_reward_mean": float(np.mean([seq_scores[idx] for idx in selected])),
        }
        return source_batch, metrics

    def _summarize_prompt_success_by_uid(
        self,
        batch: DataProto,
        positive_threshold: Optional[float] = None,
        allowed_uids: Optional[set[str]] = None,
    ) -> dict:
        if "uid" not in batch.non_tensor_batch:
            return {"num_prompts": 0, "hist": Counter(), "mean": 0.0}

        reward_key = "token_level_scores" if "token_level_scores" in batch.batch.keys() else "token_level_rewards"
        if reward_key not in batch.batch.keys():
            return {"num_prompts": 0, "hist": Counter(), "mean": 0.0}

        if positive_threshold is None:
            positive_threshold = float(self.config.algorithm.get("positive_threshold", 0.7))

        seq_scores = batch.batch[reward_key].sum(dim=-1).detach().cpu().numpy()
        uid_to_scores: dict[str, list[float]] = defaultdict(list)
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_str = str(uid)
            if allowed_uids is not None and uid_str not in allowed_uids:
                continue
            uid_to_scores[uid_str].append(float(seq_scores[idx]))

        hist = Counter()
        success_rates = []
        uid_success: dict[str, tuple[int, int]] = {}
        for uid, scores in uid_to_scores.items():
            total = len(scores)
            if total == 0:
                continue
            success = sum(score > positive_threshold for score in scores)
            hist[(success, total)] += 1
            success_rates.append(success / total)
            uid_success[uid] = (success, total)

        return {
            "num_prompts": len(success_rates),
            "hist": hist,
            "mean": float(np.mean(success_rates)) if success_rates else 0.0,
            "uid_success": uid_success,
        }

    @staticmethod
    def _format_success_hist(hist: Counter) -> str:
        if not hist:
            return "empty"
        return ", ".join(f"{success}/{total}:{count}" for (success, total), count in sorted(hist.items()))

    @staticmethod
    def _add_success_hist_metrics(metrics: dict, prefix: str, summary: dict) -> None:
        metrics[f"{prefix}/prompts"] = summary["num_prompts"]
        metrics[f"{prefix}/success_rate_mean"] = summary["mean"]
        for (success, total), count in sorted(summary["hist"].items()):
            metrics[f"{prefix}/prompts_with_{success}_of_{total}_success"] = count

    @staticmethod
    def _format_top_refine_rows(source_summary: dict, refine_summary: dict, limit: int = 10) -> list[str]:
        source_by_uid = source_summary.get("uid_success", {})
        refine_by_uid = refine_summary.get("uid_success", {})

        def sort_key(item):
            uid, (success, total) = item
            rate = success / total if total else 0.0
            return success, rate, str(uid)

        rows = []
        for rank, (uid, (src_success, src_total)) in enumerate(sorted(source_by_uid.items(), key=sort_key)[:limit], 1):
            ref_success, ref_total = refine_by_uid.get(uid, (0, 0))
            rows.append(f"  {rank:02d}. uid={uid} source={src_success}/{src_total} refine={ref_success}/{ref_total}")
        return rows

    @staticmethod
    def _format_source_bucket_refine_rows(
        source_summary: dict,
        refine_summary: dict,
        source_success: int,
        source_total: int,
        limit: int = 10,
    ) -> list[str]:
        source_by_uid = source_summary.get("uid_success", {})
        refine_by_uid = refine_summary.get("uid_success", {})
        bucket_items = [
            (uid, src_pair)
            for uid, src_pair in source_by_uid.items()
            if src_pair == (source_success, source_total)
        ]

        def sort_key(item):
            uid, _ = item
            ref_success, ref_total = refine_by_uid.get(uid, (0, 0))
            ref_rate = ref_success / ref_total if ref_total else 0.0
            return -ref_success, -ref_rate, str(uid)

        rows = []
        for rank, (uid, (src_success, src_total)) in enumerate(sorted(bucket_items, key=sort_key)[:limit], 1):
            ref_success, ref_total = refine_by_uid.get(uid, (0, 0))
            rows.append(f"  {rank:02d}. uid={uid} source={src_success}/{src_total} refine={ref_success}/{ref_total}")
        return rows

    def _build_refine_gen_batch(self, source_batch: DataProto) -> DataProto:
        raw_prompts = source_batch.non_tensor_batch.get("raw_prompt", None)
        max_prompt_length = int(self.config.algorithm.get("refine_max_prompt_length", 0) or self.config.data.max_prompt_length)
        truncation = str(self.config.algorithm.get("refine_truncation", self.config.data.truncation))
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        instruction = self.config.algorithm.get("refine_instruction", None) or DEFAULT_REFINE_INSTRUCTION

        input_ids_list = []
        attention_mask_list = []
        raw_prompt_ids_list = []
        refine_prompts = []

        for i in range(len(source_batch)):
            if raw_prompts is not None:
                question = self._extract_last_user_message(raw_prompts[i])
            else:
                prompt_ids = source_batch.batch["prompts"][i]
                prompt_ids = prompt_ids[source_batch.batch["prompts"][i] != self.tokenizer.pad_token_id]
                question = self.tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)

            previous_answer = self._decode_response_text(source_batch, i)
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": previous_answer},
                {"role": "user", "content": instruction},
            ]
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            input_ids_list.append(input_ids[0])
            attention_mask_list.append(attention_mask[0])
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
                elif truncation == "middle":
                    left_half = max_prompt_length // 2
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-(max_prompt_length - left_half):]
                elif truncation == "error":
                    raise RuntimeError(f"Refine prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length=}.")
            raw_prompt_ids_list.append(raw_prompt_ids)
            refine_prompts.append(messages)

        attention_mask = torch.stack(attention_mask_list, dim=0)
        tensors = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        }
        non_tensors = {}
        for key in ("data_source", "reward_model", "uid"):
            if key in source_batch.non_tensor_batch:
                non_tensors[key] = source_batch.non_tensor_batch[key]

        extra_info = source_batch.non_tensor_batch.get("extra_info", np.array([{}] * len(source_batch), dtype=object))
        updated_extra = []
        for item in extra_info.tolist() if isinstance(extra_info, np.ndarray) else list(extra_info):
            new_item = dict(item) if isinstance(item, dict) else {"value": item}
            new_item["is_refine"] = True
            updated_extra.append(new_item)
        non_tensors["extra_info"] = np.array(updated_extra, dtype=object)
        non_tensors["raw_prompt"] = np.array(refine_prompts, dtype=object)
        non_tensors["raw_prompt_ids"] = np.array(raw_prompt_ids_list, dtype=object)

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(source_batch.meta_info))

    @staticmethod
    def _token_scores_from_sequence_rewards_like(batch: DataProto, seq_rewards: list[float]) -> torch.Tensor:
        token_scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        response_mask = batch.batch["response_mask"]
        for i, reward in enumerate(seq_rewards):
            valid = torch.nonzero(response_mask[i].bool(), as_tuple=False).flatten()
            if len(valid) > 0:
                token_scores[i, valid[-1]] = float(reward)
        return token_scores

    def _select_rewrite_source_batch(
        self,
        refine_batch: DataProto,
        source_success_summary: dict,
    ) -> tuple[Optional[DataProto], dict[str, float]]:
        cfg = self.config.algorithm
        positive_threshold = float(cfg.get("positive_threshold", 0.7))
        source_bsz = int(cfg.get("rewrite_source_bsz", 0) or self.config.data.train_batch_size)
        difficult_space = int(cfg.get("difficult_rewrite_space", 128) or 0)
        seed = int(self.config.data.get("seed", 0) or 0) + int(self.global_steps) + 9001
        rng = random.Random(seed)

        if "uid" not in refine_batch.non_tensor_batch or "token_level_scores" not in refine_batch.batch.keys():
            return None, {"rewrite/source_refines": 0}

        seq_scores = refine_batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
        uid_to_correct: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(refine_batch.non_tensor_batch["uid"]):
            if seq_scores[idx] > positive_threshold:
                uid_to_correct[str(uid)].append(idx)

        source_by_uid = source_success_summary.get("uid_success", {})
        difficult_order = sorted(
            source_by_uid.keys(),
            key=lambda uid: (
                source_by_uid[uid][0] / max(source_by_uid[uid][1], 1),
                source_by_uid[uid][0],
                str(uid),
            ),
        )

        selected: list[int] = []
        selected_set: set[int] = set()
        difficult_selected = 0
        for uid in difficult_order[: max(0, difficult_space)]:
            candidates = [idx for idx in uid_to_correct.get(str(uid), []) if idx not in selected_set]
            if not candidates:
                continue
            idx = rng.choice(candidates)
            selected.append(idx)
            selected_set.add(idx)
            difficult_selected += 1
            if len(selected) >= source_bsz:
                break

        if len(selected) < source_bsz:
            remaining = [
                idx
                for indices in uid_to_correct.values()
                for idx in indices
                if idx not in selected_set
            ]
            rng.shuffle(remaining)
            selected.extend(remaining[: source_bsz - len(selected)])

        if not selected:
            return None, {"rewrite/source_refines": 0}

        selected_batch = refine_batch[selected]
        metrics = {
            "rewrite/source_refines": len(selected),
            "rewrite/source_refines_available": sum(len(v) for v in uid_to_correct.values()),
            "rewrite/difficult_space": difficult_space,
            "rewrite/difficult_selected": difficult_selected,
        }
        return selected_batch, metrics

    def _build_rewrite_gen_batch(self, source_batch: DataProto) -> DataProto:
        raw_prompts = source_batch.non_tensor_batch.get("raw_prompt", None)
        max_prompt_length = int(
            self.config.algorithm.get("rewrite_max_prompt_length", 0)
            or self.config.algorithm.get("refine_max_prompt_length", 0)
            or self.config.data.max_prompt_length
        )
        truncation = str(self.config.algorithm.get("rewrite_truncation", self.config.data.truncation))
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        instruction = self.config.algorithm.get("rewrite_instruction", None) or DEFAULT_REWRITE_INSTRUCTION

        input_ids_list = []
        attention_mask_list = []
        raw_prompt_ids_list = []
        rewrite_prompts = []
        rewrite_questions = []
        rewrite_references = []
        rewrite_refined = []

        for i in range(len(source_batch)):
            raw_prompt = raw_prompts[i] if raw_prompts is not None else ""
            question = self._extract_role_message(raw_prompt, role="user", first=True)
            previous_solution = self._extract_role_message(raw_prompt, role="assistant", first=True)
            refined_solution = self._decode_response_text(source_batch, i)
            content = (
                "Your previous solution was:\n"
                f"{previous_solution}\n\n"
                "Your review/refined solution was:\n"
                f"{refined_solution}\n\n"
                f"{instruction}"
            )
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": refined_solution},
                {"role": "user", "content": content},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            model_inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            raw_prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                raw_prompt_ids = raw_prompt_ids[-max_prompt_length:] if truncation == "left" else raw_prompt_ids[:max_prompt_length]
            input_ids_list.append(input_ids[0])
            attention_mask_list.append(attention_mask[0])
            raw_prompt_ids_list.append(raw_prompt_ids)
            rewrite_prompts.append(messages)
            rewrite_questions.append(question)
            rewrite_references.append(refined_solution)
            rewrite_refined.append(refined_solution)

        attention_mask = torch.stack(attention_mask_list, dim=0)
        tensors = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        }
        non_tensors = {}
        for key in ("data_source", "reward_model", "uid"):
            if key in source_batch.non_tensor_batch:
                non_tensors[key] = source_batch.non_tensor_batch[key]
        non_tensors["raw_prompt"] = np.array(rewrite_prompts, dtype=object)
        non_tensors["raw_prompt_ids"] = np.array(raw_prompt_ids_list, dtype=object)
        non_tensors["rewrite_question"] = np.array(rewrite_questions, dtype=object)
        non_tensors["rewrite_reference_text"] = np.array(rewrite_references, dtype=object)
        non_tensors["rewrite_refined_text"] = np.array(rewrite_refined, dtype=object)
        extra_info = source_batch.non_tensor_batch.get("extra_info", np.array([{}] * len(source_batch), dtype=object))
        updated_extra = []
        for item in extra_info.tolist() if isinstance(extra_info, np.ndarray) else list(extra_info):
            new_item = dict(item) if isinstance(item, dict) else {"value": item}
            new_item["is_rewrite"] = True
            updated_extra.append(new_item)
        non_tensors["extra_info"] = np.array(updated_extra, dtype=object)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(source_batch.meta_info))

    def _build_synthetic_batch_from_rewrites(self, rewrite_batch: DataProto, indices: list[int]) -> Optional[DataProto]:
        if not indices:
            return None
        selected = rewrite_batch[indices]
        max_prompt_length = int(self.config.data.max_prompt_length)
        max_response_length = int(self.config.data.max_response_length)
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        prompt_ids_list = []
        prompt_mask_list = []
        response_ids_list = []
        response_mask_list = []
        raw_prompts = []
        raw_prompt_ids = []

        for i in range(len(selected)):
            question = str(selected.non_tensor_batch.get("rewrite_question", [""] * len(selected))[i])
            messages = [{"role": "user", "content": question}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            model_inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            prompt_ids, prompt_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            resp = selected.batch["responses"][i]
            resp_mask = selected.batch["response_mask"][i].bool()
            resp = resp[resp_mask][:max_response_length]
            padded_resp = torch.full((max_response_length,), self.tokenizer.pad_token_id, dtype=resp.dtype)
            padded_resp_mask = torch.zeros((max_response_length,), dtype=prompt_mask.dtype)
            if len(resp) > 0:
                padded_resp[: len(resp)] = resp
                padded_resp_mask[: len(resp)] = 1
            prompt_ids_list.append(prompt_ids[0])
            prompt_mask_list.append(prompt_mask[0])
            response_ids_list.append(padded_resp)
            response_mask_list.append(padded_resp_mask)
            raw_prompts.append(messages)
            encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            raw_prompt_ids.append(encoded_prompt[-max_prompt_length:])

        prompts = torch.stack(prompt_ids_list, dim=0)
        responses = torch.stack(response_ids_list, dim=0)
        prompt_attention = torch.stack(prompt_mask_list, dim=0)
        response_mask = torch.stack(response_mask_list, dim=0)
        attention_mask = torch.cat([prompt_attention, response_mask], dim=-1)
        tensors = {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=-1),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "response_mask": response_mask,
        }
        non_tensors = {}
        for key in ("data_source", "reward_model", "uid"):
            if key in selected.non_tensor_batch:
                non_tensors[key] = selected.non_tensor_batch[key]
        non_tensors["raw_prompt"] = np.array(raw_prompts, dtype=object)
        non_tensors["raw_prompt_ids"] = np.array(raw_prompt_ids, dtype=object)
        non_tensors["extra_info"] = np.array([{"is_synthetic_rewrite": True} for _ in range(len(selected))], dtype=object)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(selected.meta_info))

    def _run_rewrite_steps(
        self,
        refine_batch: DataProto,
        source_success_summary: dict,
        epoch: int,
        logger,
        progress_bar,
    ) -> bool:
        if not self.config.algorithm.get("enable_rewrite", False):
            return False
        if not self.config.algorithm.get("enable_refine", False):
            raise ValueError("algorithm.enable_rewrite=True requires algorithm.enable_refine=True")
        if self.global_steps > self.total_training_steps:
            return True

        source_batch, rewrite_metrics = self._select_rewrite_source_batch(refine_batch, source_success_summary)
        if source_batch is None or len(source_batch) == 0:
            print("[Rewrite] No successful refine source selected; skip rewrite steps.")
            return False

        rewrite_repeat = int(self.config.algorithm.get("rewrite_per_prompt", 4) or 4)
        source_original_len = len(source_batch)
        world_size = self.actor_rollout_wg.world_size
        if source_original_len * rewrite_repeat % world_size != 0:
            padding_needed = 1
            while (source_original_len + padding_needed) * rewrite_repeat % world_size != 0:
                padding_needed += 1
            padding_batch = source_batch[random.choices(range(source_original_len), k=padding_needed)]
            source_batch = DataProto.concat([source_batch, padding_batch])
            print(
                f"[Rewrite] Padding source refines from {source_original_len} "
                f"to {len(source_batch)} for repeat={rewrite_repeat}, dp_world_size={world_size}"
            )

        timing_raw = {}
        metrics = dict(rewrite_metrics)
        metrics.update(
            {
                "rewrite/source_refines_original": source_original_len,
                "rewrite/source_refines_padded": len(source_batch),
                "rewrite/source_refines_padding": len(source_batch) - source_original_len,
            }
        )
        correct_flags: list[bool] = []
        composite_rewards: list[float] = []
        rewrite_batch = None

        with marked_timer("step", timing_raw):
            with marked_timer("rewrite_build_prompt", timing_raw, color="green"):
                rewrite_prompt_batch = self._build_rewrite_gen_batch(source_batch)
                gen_batch = self._get_gen_batch(rewrite_prompt_batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=rewrite_repeat, interleave=True)

            with marked_timer("rewrite_gen", timing_raw, color="red"):
                gen_batch_output = (
                    self.actor_rollout_wg.generate_sequences(gen_batch)
                    if not self.async_rollout_mode
                    else self.async_rollout_manager.generate_sequences(gen_batch)
                )
                timing_raw.update({f"rewrite/{k}": v for k, v in gen_batch_output.meta_info.get("timing", {}).items()})
                gen_batch_output.meta_info.pop("timing", None)

            rewrite_batch = rewrite_prompt_batch.repeat(repeat_times=rewrite_repeat, interleave=True)
            rewrite_batch = rewrite_batch.union(gen_batch_output)
            if "response_mask" not in rewrite_batch.batch.keys():
                rewrite_batch.batch["response_mask"] = compute_response_mask(rewrite_batch)

            if self.config.trainer.balance_batch:
                world_size = self.actor_rollout_wg.world_size
                batch_size = len(rewrite_batch)
                if batch_size % world_size != 0:
                    padding_needed = world_size - (batch_size % world_size)
                    padding_batch = rewrite_batch[random.choices(range(batch_size), k=padding_needed)]
                    rewrite_batch = DataProto.concat([rewrite_batch, padding_batch])
                self._balance_batch(rewrite_batch, metrics=metrics, logging_prefix="rewrite_global_seqlen")
            rewrite_batch.batch = rewrite_batch.batch.contiguous()
            rewrite_batch.meta_info["global_token_num"] = torch.sum(rewrite_batch.batch["attention_mask"], dim=-1).tolist()

            with marked_timer("rewrite_reward", timing_raw, color="yellow"):
                correctness_tensor, reward_extra_infos_dict = compute_reward(rewrite_batch, self.reward_fn)

            seq_correct_scores = correctness_tensor.sum(dim=-1).detach().cpu().numpy()
            positive_threshold = float(self.config.algorithm.get("positive_threshold", 0.7))
            references = rewrite_batch.non_tensor_batch.get("rewrite_reference_text", np.array([""] * len(rewrite_batch), dtype=object))
            refined_texts = rewrite_batch.non_tensor_batch.get("rewrite_refined_text", references)
            reward_infos = []
            for i in range(len(rewrite_batch)):
                rewrite_text = self._decode_response_text(rewrite_batch, i)
                correct = bool(seq_correct_scores[i] > positive_threshold)
                correct_flags.append(correct)
                info = _rewrite_composite_reward(
                    correct=correct,
                    reference=str(references[i]),
                    refined=str(refined_texts[i]),
                    rewrite=rewrite_text,
                    rouge_target=float(self.config.algorithm.get("rewrite_rouge_target", 0.50)),
                    rouge_width=float(self.config.algorithm.get("rewrite_rouge_width", 0.20)),
                    length_width=float(self.config.algorithm.get("rewrite_length_width", 0.75)),
                    length_low_scale=float(self.config.algorithm.get("rewrite_length_low_scale", 0.40)),
                    length_high_scale=float(self.config.algorithm.get("rewrite_length_high_scale", 1.50)),
                )
                reward_infos.append(info)
                composite_rewards.append(float(info["reward"]))

            rewrite_scores = self._token_scores_from_sequence_rewards_like(rewrite_batch, composite_rewards)
            rewrite_batch.batch["token_level_scores"] = rewrite_scores
            rewrite_batch.batch["token_level_rewards"] = rewrite_scores
            if reward_extra_infos_dict:
                rewrite_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            metrics.update(
                {
                    "rewrite/trajectories": len(rewrite_batch),
                    "rewrite/correct_rate": float(np.mean(correct_flags)) if correct_flags else 0.0,
                    "rewrite/composite_reward_mean": float(np.mean(composite_rewards)) if composite_rewards else 0.0,
                    "rewrite/composite_reward_max": float(np.max(composite_rewards)) if composite_rewards else 0.0,
                    "rewrite/rouge_mix_mean": float(np.mean([x["rouge_mix"] for x in reward_infos])) if reward_infos else 0.0,
                    "rewrite/length_score_mean": float(np.mean([x["length_score"] for x in reward_infos])) if reward_infos else 0.0,
                }
            )

            with marked_timer("rewrite_old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(rewrite_batch)
                entropys = old_log_prob.batch["entropys"]
                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=rewrite_batch.batch["response_mask"],
                    loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                )
                metrics["rewrite/actor_entropy"] = entropy_agg.detach().item()
                old_log_prob.batch.pop("entropys")
                rewrite_batch = rewrite_batch.union(old_log_prob)

            if self.use_reference_policy:
                with marked_timer("rewrite_ref", timing_raw, color="olive"):
                    ref_log_prob = (
                        self.ref_policy_wg.compute_ref_log_prob(rewrite_batch)
                        if not self.ref_in_actor
                        else self.actor_rollout_wg.compute_ref_log_prob(rewrite_batch)
                    )
                    rewrite_batch = rewrite_batch.union(ref_log_prob)

            with marked_timer("rewrite_adv", timing_raw, color="brown"):
                rewrite_algo_config = OmegaConf.create(OmegaConf.to_container(self.config.algorithm, resolve=True))
                rewrite_algo_config.global_stat_est = False
                rewrite_algo_config.correct_bias = False
                batch_for_adv = compute_advantage(
                    rewrite_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=rewrite_repeat,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=rewrite_algo_config,
                )
                rewrite_batch = batch_for_adv

            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("rewrite_update_actor", timing_raw, color="red"):
                    rewrite_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(rewrite_batch)
                metrics.update({f"rewrite/{k}": v for k, v in reduce_metrics(actor_output.meta_info["metrics"]).items()})

        is_last_step = self.global_steps >= self.total_training_steps
        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
        ):
            with marked_timer("rewrite_testing", timing_raw, color="green"):
                metrics.update(self._validate())

        metrics.update(
            {
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
                "training/is_refine_step": 0,
                "training/is_rewrite_step": 1,
                "training/is_synthetic_step": 0,
            }
        )
        metrics.update({f"rewrite/{k}": v for k, v in compute_data_metrics(batch=rewrite_batch, use_critic=False).items()})
        metrics.update({f"rewrite/{k}": v for k, v in compute_timing_metrics(batch=rewrite_batch, timing_raw=timing_raw).items()})
        print(
            f"[Rewrite] step={self.global_steps} source_refines={len(source_batch)} "
            f"repeat={rewrite_repeat} trajectories={len(rewrite_batch)} "
            f"correct_rate={metrics['rewrite/correct_rate']:.4f} "
            f"reward_mean={metrics['rewrite/composite_reward_mean']:.4f}"
        )
        logger.log(data=metrics, step=self.global_steps)
        progress_bar.update(1)
        self.global_steps += 1

        if self.global_steps > self.total_training_steps:
            return True

        correct_indices = [idx for idx, correct in enumerate(correct_flags) if correct]
        synthetic_batch = self._build_synthetic_batch_from_rewrites(rewrite_batch, correct_indices)
        if synthetic_batch is None or len(synthetic_batch) == 0:
            print("[Synthetic-Rewrite] No correct rewrite samples; skip synthetic update.")
            return False

        synthetic_original_len = len(synthetic_batch)
        world_size = self.actor_rollout_wg.world_size
        if synthetic_original_len % world_size != 0:
            padding_needed = world_size - (synthetic_original_len % world_size)
            padding_batch = synthetic_batch[random.choices(range(synthetic_original_len), k=padding_needed)]
            synthetic_batch = DataProto.concat([synthetic_batch, padding_batch])
            print(
                f"[Synthetic-Rewrite] Padding synthetic batch from {synthetic_original_len} "
                f"to {len(synthetic_batch)} for dp_world_size={world_size}"
            )

        synth_timing = {}
        synth_metrics = {
            "synthetic_rewrite/source_correct_rewrites": len(correct_indices),
            "synthetic_rewrite/original_trajectories": synthetic_original_len,
            "synthetic_rewrite/trajectories": len(synthetic_batch),
            "synthetic_rewrite/padding": len(synthetic_batch) - synthetic_original_len,
        }
        with marked_timer("step", synth_timing):
            synthetic_batch.batch = synthetic_batch.batch.contiguous()
            synthetic_batch.meta_info["global_token_num"] = torch.sum(synthetic_batch.batch["attention_mask"], dim=-1).tolist()
            with marked_timer("synthetic_old_log_prob", synth_timing, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(synthetic_batch)
                if "entropys" in old_log_prob.batch.keys():
                    old_log_prob.batch.pop("entropys")
                synthetic_batch = synthetic_batch.union(old_log_prob)

            seq_rewards = [1.0] * len(synthetic_batch)
            token_scores = self._token_scores_from_sequence_rewards_like(synthetic_batch, seq_rewards)
            synthetic_batch.batch["token_level_scores"] = token_scores
            synthetic_batch.batch["token_level_rewards"] = token_scores
            synthetic_batch.batch["advantages"] = synthetic_batch.batch["response_mask"].float()
            synthetic_batch.batch["returns"] = synthetic_batch.batch["advantages"]

            with marked_timer("synthetic_update_actor", synth_timing, color="red"):
                synthetic_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(synthetic_batch)
            synth_metrics.update(
                {f"synthetic_rewrite/{k}": v for k, v in reduce_metrics(actor_output.meta_info["metrics"]).items()}
            )

        is_last_step = self.global_steps >= self.total_training_steps
        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
        ):
            with marked_timer("synthetic_testing", synth_timing, color="green"):
                synth_metrics.update(self._validate())

        synth_metrics.update(
            {
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
                "training/is_refine_step": 0,
                "training/is_rewrite_step": 0,
                "training/is_synthetic_step": 1,
            }
        )
        synth_metrics.update(
            {f"synthetic_rewrite/{k}": v for k, v in compute_timing_metrics(batch=synthetic_batch, timing_raw=synth_timing).items()}
        )
        print(
            f"[Synthetic-Rewrite] step={self.global_steps} trajectories={len(synthetic_batch)} "
            f"source_correct_rewrites={len(correct_indices)}"
        )
        debug_text = self._decode_response_text(synthetic_batch, 0) if len(synthetic_batch) > 0 else ""
        debug_rewrite_idx = correct_indices[0] if correct_indices else None
        debug_reward_info = reward_infos[debug_rewrite_idx] if debug_rewrite_idx is not None else {}
        debug_correct = correct_flags[debug_rewrite_idx] if debug_rewrite_idx is not None else False
        debug_ground_truth = ""
        if len(synthetic_batch) > 0 and "reward_model" in synthetic_batch.non_tensor_batch:
            reward_model_item = synthetic_batch.non_tensor_batch["reward_model"][0]
            if isinstance(reward_model_item, dict):
                debug_ground_truth = str(reward_model_item.get("ground_truth", ""))
        print(
            "[Synthetic-Rewrite-Debug-Sample] "
            f"correct={debug_correct} "
            "judge_scope=response_only "
            f"reward={float(debug_reward_info.get('reward', 0.0)):.4f} "
            f"rouge_mix={float(debug_reward_info.get('rouge_mix', 0.0)):.4f} "
            f"rouge_score={float(debug_reward_info.get('rouge_score', 0.0)):.4f} "
            f"length_score={float(debug_reward_info.get('length_score', 0.0)):.4f} "
            f"rewrite_len={int(debug_reward_info.get('rewrite_len', 0))} "
            f"ground_truth={debug_ground_truth}"
            f"\n{debug_text[:2000]}"
        )
        logger.log(data=synth_metrics, step=self.global_steps)
        progress_bar.update(1)
        self.global_steps += 1
        return self.global_steps > self.total_training_steps

    def _run_refine_step(self, source_final_batch: DataProto, epoch: int, logger, progress_bar) -> bool:
        if self.global_steps > self.total_training_steps:
            return False

        source_batch, refine_metrics = self._select_refine_source_batch(source_final_batch)
        if source_batch is None or len(source_batch) == 0:
            print("[Refine] No source batch selected; skip refine step.")
            return False

        metrics = dict(refine_metrics)
        source_uids = {str(uid) for uid in source_batch.non_tensor_batch.get("uid", [])}
        source_success_summary = self._summarize_prompt_success_by_uid(
            source_final_batch,
            allowed_uids=source_uids,
        )
        self._add_success_hist_metrics(metrics, "refine/source_prompt_success", source_success_summary)
        timing_raw = {}
        is_last_step = self.global_steps >= self.total_training_steps
        refine_repeat = int(self.config.algorithm.get("refine_per_prompt", 4) or 4)
        refine_success_summary = {"num_prompts": 0, "hist": Counter(), "mean": 0.0}

        with marked_timer("step", timing_raw):
            with marked_timer("refine_build_prompt", timing_raw, color="green"):
                refine_prompt_batch = self._build_refine_gen_batch(source_batch)
                gen_batch = self._get_gen_batch(refine_prompt_batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=refine_repeat, interleave=True)

            with marked_timer("refine_gen", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                timing_raw.update({f"refine/{k}": v for k, v in gen_batch_output.meta_info.get("timing", {}).items()})
                gen_batch_output.meta_info.pop("timing", None)

            batch = refine_prompt_batch.repeat(repeat_times=refine_repeat, interleave=True)
            batch = batch.union(gen_batch_output)

            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)

            if self.config.trainer.balance_batch:
                world_size = self.actor_rollout_wg.world_size
                batch_size = len(batch)
                if batch_size % world_size != 0:
                    padding_needed = world_size - (batch_size % world_size)
                    padding_batch = batch[random.choices(range(batch_size), k=padding_needed)]
                    batch = DataProto.concat([batch, padding_batch])
                self._balance_batch(batch, metrics=metrics)
            batch.batch = batch.batch.contiguous()
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with marked_timer("refine_reward", timing_raw, color="yellow"):
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

            with marked_timer("refine_old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=batch.batch["response_mask"],
                    loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                )
                metrics["refine/actor_entropy"] = entropy_agg.detach().item()
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                with marked_timer("refine_ref", timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            if self.use_critic:
                with marked_timer("refine_values", timing_raw, color="cyan"):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            with marked_timer("refine_adv", timing_raw, color="brown"):
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor
                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                refine_success_summary = self._summarize_prompt_success_by_uid(batch)
                self._add_success_hist_metrics(metrics, "refine/result_prompt_success", refine_success_summary)

                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update({f"refine/{k}": v for k, v in kl_metrics.items()})
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                refine_algo_config = OmegaConf.create(OmegaConf.to_container(self.config.algorithm, resolve=True))
                refine_algo_config.global_stat_est = False
                refine_algo_config.correct_bias = False
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=refine_repeat,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=refine_algo_config,
                )

            if self.use_critic:
                with marked_timer("refine_update_critic", timing_raw, color="pink"):
                    critic_output = self.critic_wg.update_critic(batch)
                metrics.update({f"refine/{k}": v for k, v in reduce_metrics(critic_output.meta_info["metrics"]).items()})

            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("refine_update_actor", timing_raw, color="red"):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics.update({f"refine/{k}": v for k, v in reduce_metrics(actor_output.meta_info["metrics"]).items()})

        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
        ):
            with marked_timer("refine_testing", timing_raw, color="green"):
                metrics.update(self._validate())

        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        if self.config.trainer.save_freq > 0 and (
            is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            with marked_timer("refine_save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()

        steps_duration = timing_raw["step"]
        self.max_steps_duration = max(self.max_steps_duration, steps_duration)
        metrics.update(
            {
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
                "training/is_refine_step": 1,
            }
        )
        metrics.update({f"refine/{k}": v for k, v in compute_data_metrics(batch=batch, use_critic=self.use_critic).items()})
        metrics.update({f"refine/{k}": v for k, v in compute_timing_metrics(batch=batch, timing_raw=timing_raw).items()})
        metrics.update(
            {
                f"refine/{k}": v
                for k, v in compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, n_gpus=self.resource_pool_manager.get_n_gpus()
                ).items()
            }
        )

        print(
            f"[Refine] step={self.global_steps} source_prompts={len(source_batch)} "
            f"repeat={refine_repeat} trajectories={len(batch)}"
        )
        print(
            f"[Refine-Source] prompts={source_success_summary['num_prompts']} "
            f"success_mean={source_success_summary['mean']:.4f} "
            f"success_hist={self._format_success_hist(source_success_summary['hist'])}"
        )
        print(
            f"[Refine-Result] prompts={refine_success_summary['num_prompts']} "
            f"refine_per_prompt={refine_repeat} "
            f"success_mean={refine_success_summary['mean']:.4f} "
            f"success_hist={self._format_success_hist(refine_success_summary['hist'])}"
        )
        print("[Refine-Top10-Hardest]")
        top_refine_rows = self._format_top_refine_rows(source_success_summary, refine_success_summary, limit=10)
        for row in top_refine_rows:
            print(row)
        if not top_refine_rows:
            print("  empty")
        print("[Refine-Top10-Source-1of4]")
        one_of_four_rows = self._format_source_bucket_refine_rows(
            source_success_summary,
            refine_success_summary,
            source_success=1,
            source_total=4,
            limit=10,
        )
        for row in one_of_four_rows:
            print(row)
        if not one_of_four_rows:
            print("  empty")
        logger.log(data=metrics, step=self.global_steps)
        progress_bar.update(1)
        self.global_steps += 1
        if (
            self.config.algorithm.get("enable_rewrite", False)
            and self.global_steps <= self.total_training_steps
        ):
            stop_after_rewrite = self._run_rewrite_steps(batch, source_success_summary, epoch, logger, progress_bar)
            return is_last_step or stop_after_rewrite
        return is_last_step

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    if self.config.algorithm.multiround_adaptive_downsampling:
                        with marked_timer("gen_multi_round", timing_raw, color="red"):
                            final_batch, rounds_info = self._generate_multi_round_adaptive_downsampling(
                                orig_prompt_batch=gen_batch,
                                positive_threshold=self.config.algorithm.positive_threshold,
                                max_rounds=self.config.algorithm.max_rounds,
                                round_repeat=self.config.algorithm.round_repeat,
                                final_keep_per_prompt=self.config.actor_rollout_ref.rollout.n,
                                timing_raw=timing_raw,
                                context_batch=batch,
                            )

                        total_prompts = len(set(gen_batch.non_tensor_batch["uid"]))
                        print(
                            f"[Summary] prompts={total_prompts}, selected_rows={len(final_batch)}, "
                            f"max_rounds={self.config.algorithm.max_rounds}"
                        )
                        if rounds_info.get("per_round"):
                            for info in rounds_info["per_round"]:
                                print(
                                    f"  - round {info['round']}: active={info['active_prompts']}, "
                                    f"completed={info['completed']}, finished={info['finished_prompts']}, "
                                    f"time={info['sec']}s"
                                )

                        metrics["sampling/total_samples"] = np.sum(
                            [
                                (info["active_prompts"] * self.config.algorithm.round_repeat)
                                for info in rounds_info["per_round"]
                            ]
                        )
                        metrics["sampling/prompts_active_only_1st_round"] = rounds_info["per_round"][0][
                            "finished_prompts"
                        ]

                        if len(rounds_info["per_round"]) > 1:
                            metrics["sampling/prompts_active_after_1st_round"] = rounds_info["per_round"][1][
                                "active_prompts"
                            ] - (
                                rounds_info["per_round"][0]["active_prompts"]
                                - rounds_info["per_round"][-1]["finished_prompts"]
                            )
                        else:
                            metrics["sampling/prompts_active_after_1st_round"] = 0

                        metrics["sampling/prompts_no_positive_anywhere"] = (
                            rounds_info["per_round"][0]["active_prompts"]
                            - rounds_info["per_round"][-1]["finished_prompts"]
                        )
                        metrics["sampling/kept_samples"] = len(final_batch)
                        metrics["critic/real_reward"] = rounds_info["per_round"][0]["reward_mean"]
                        metrics["sampling/downsampled_samples"] = len(final_batch)
                        metrics["sampling/total_prompts"] = total_prompts

                        batch = final_batch

                    else:
                        with marked_timer("gen", timing_raw, color="red"):
                            gen_batch = gen_batch.repeat(
                                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            )

                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                        gen_baseline_batch
                                    )
                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del gen_baseline_batch, gen_baseline_output

                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    if self.config.trainer.balance_batch:
                        world_size = self.actor_rollout_wg.world_size
                        batch_size = len(batch)
                        if batch_size % world_size == 0:
                            self._balance_batch(batch, metrics=metrics)
                        else:
                            # Pad the batch to make it divisible by world_size
                            padding_needed = world_size - (batch_size % world_size)
                            print(f"Padding batch from {batch_size} to {batch_size + padding_needed} for balancing")

                            indices_to_repeat = random.choices(range(batch_size), k=padding_needed)
                            padding_batch = batch[indices_to_repeat]
                            batch = DataProto.concat([batch, padding_batch])

                            if hasattr(batch.batch, "__class__"):
                                batch_type = batch.batch.__class__.__name__
                                if "TensorDict" not in batch_type and "dict" in batch_type.lower():
                                    print(
                                        f"[perf_warn] After padding batch.batch is plain {batch_type}, may affect performance"
                                    )

                            self._balance_batch(batch, metrics=metrics)
                    batch.batch = batch.batch.contiguous()

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # reward processing and downsampling already done in multi-round generation
                        if not self.config.algorithm.multiround_adaptive_downsampling:
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/is_refine_step": 0,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if self.config.algorithm.get("enable_refine", False):
                    self._last_final_batch = batch

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if (
                    self.config.algorithm.get("enable_refine", False)
                    and hasattr(self, "_last_final_batch")
                    and self._last_final_batch is not None
                    and self.global_steps <= self.total_training_steps
                ):
                    stop_after_refine = self._run_refine_step(self._last_final_batch, epoch, logger, progress_bar)
                    self._last_final_batch = None
                    if stop_after_refine:
                        progress_bar.close()
                        return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
