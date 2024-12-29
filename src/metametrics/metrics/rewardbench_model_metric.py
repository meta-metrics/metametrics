# Copyright 2023 AllenAI. All rights reserved.
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

# Run RewardBench (evaluate any reward model on any dataet)

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoTokenizer

from rewardbench import (
    DPO_MODEL_CONFIG,
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_preference_dataset,
)

from metametrics.metrics.base_metric import RewardBaseMetric
from metametrics.utils.validate import validate_argument_list, validate_int, validate_real, validate_bool

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class RewardBenchArgs:
    # core args
    model: Optional[str] = None
    """The model to evaluate."""
    revision: Optional[str] = None
    """The model revision to evaluate."""
    ref_model: Optional[str] = None
    """The reference model to compare against."""
    tokenizer: Optional[str] = None
    """The tokenizer to use (defaults to model)."""
    chat_template: Optional[str] = None
    """The chat template to use (defaults to from tokenizer, from chattemplate)."""
    not_quantized: bool = False
    """Disable quantization for models that are quantized by default."""

    # wandb args
    wandb_run: Optional[str] = None
    """The wandb run to extract model and revision from."""

    # inference args
    batch_size: int = 8
    """The batch size to use."""
    max_length: int = 512
    """The max length to use."""

    # system args
    load_json: bool = False
    """Load dataset as json."""
    trust_remote_code: bool = False
    """Trust remote code."""
    force_truncation: bool = False
    """Force truncation (for if model errors)."""
    
class RewardBenchModelMetric(RewardBaseMetric):
    def __init__(self, args: Dict[str, Any]):
        self.args = RewardBenchArgs(**args)
        
    def _evaluate_reward(self, dataset_name, split=None, max_samples=None):
        if self.args.wandb_run is not None:
            wandb_run = wandb.Api().run(self.args.wandb_run)
            self.args.model = wandb_run.config["hf_repo_id"]
            self.args.revision = wandb_run.config["hf_repo_revision"]

        ###############
        # Setup logging
        ###############
        accelerator = Accelerator()
        current_device = accelerator.process_index

        logger.info(f"Running reward model on {self.args.model} with chat template {self.args.chat_template}")
        if self.args.trust_remote_code:
            logger.info("Loading model with Trust Remote Code")

        # basic checks from config
        if self.args.ref_model:
            is_dpo = True
            MODEL_CONFIGS = DPO_MODEL_CONFIG
            assert self.args.model != self.args.ref_model, "policy and reference model should be different"
            from trl.trainer.utils import DPODataCollatorWithPadding

            from rewardbench import DPOInference
        else:
            is_dpo = False
            MODEL_CONFIGS = REWARD_MODEL_CONFIG

        if self.args.chat_template:
            from fastchat.conversation import get_conv_template

            conv = get_conv_template(self.args.chat_template)
        else:
            conv = None

        if self.args.model in MODEL_CONFIGS:
            config = MODEL_CONFIGS[self.args.model]
        else:
            config = MODEL_CONFIGS["default"]
        logger.info(f"Using reward model config: {config}")

        # Default entries
        # "model_builder": AutoModelForSequenceClassification.from_pretrained,
        # "pipeline_builder": pipeline,
        # "quantized": True,
        # "custom_dialogue": False,
        # "model_type": "Seq. Classifier"

        if not is_dpo:
            quantized = config["quantized"]  # only Starling isn't quantized for now
            # if llama-3 in name, switch quantized to False (severely degrades performance)
            if (
                ("llama-3" in self.args.model)
                or ("Llama3" in self.args.model)
                or ("Llama-3" in self.args.model)
                or ("LLaMA3" in self.args.model)
                or self.args.not_quantized
            ):
                quantized = False
                logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {self.args.not_quantized})")
            custom_dialogue = config["custom_dialogue"]
            pipeline_builder = config["pipeline_builder"]
            _ = config["model_type"]
            if custom_dialogue:
                raise NotImplementedError("Custom dialogue not implemented yet for simpler data formatting.")

        model_builder = config["model_builder"]

        #########################
        # load dataset
        #########################
        logger.info("*** Load dataset ***")
        tokenizer_path = self.args.tokenizer if self.args.tokenizer else self.args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=self.args.trust_remote_code, revision=self.args.revision
        )
        if dataset_name == "allenai/reward-bench":
            logger.info("Running core eval dataset.")
            from rewardbench import load_eval_dataset
            from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
            from rewardbench.utils import calculate_scores_per_section

            # primary set compiles slightly more information
            dataset, subsets = load_eval_dataset(
                core_set=True,
                conv=conv,
                custom_dialogue_formatting=False,
                tokenizer=tokenizer,
                logger=logger,
                keep_columns=["text_chosen", "text_rejected", "prompt"],
            )
            self.subsets = subsets
        else:
            dataset = load_preference_dataset(
                dataset_name, split=split, json=self.args.load_json, tokenizer=tokenizer, conv=conv
            )
        
        if max_samples is not None:
            self.dataset = dataset.select(range(max_samples))
        else:
            self.dataset = dataset

        logger.info("*** Load reward model ***")

        ############################
        # Load DPO model pipeline
        ############################
        if is_dpo:
            tokenizer.pad_token = tokenizer.eos_token
            # if no BOS token, set as pad token, e.g. QWEN models
            if tokenizer.bos_token is None:
                tokenizer.bos_token_id = tokenizer.eos_token_id
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model_kwargs = {
                "load_in_8bit": True,
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
            }
            model = model_builder(
                self.args.model,
                trust_remote_code=self.args.trust_remote_code,
                **model_kwargs,
            )
            ref_model = model_builder(
                self.args.ref_model,
                trust_remote_code=self.args.trust_remote_code,
                **model_kwargs,
            )

            # use internal inference functions in DPO trainer
            dpo = DPOInference(
                model,
                ref_model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                # norm is norm, avg is average, sum is sum
            )

            # tokenize dataset
            column_names = list(dataset.features)

            tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)
            dataloader = torch.utils.data.DataLoader(
                tokenized_dataset,
                batch_size=self.args.batch_size,
                collate_fn=DPODataCollatorWithPadding(
                    pad_token_id=tokenizer.pad_token_id,
                    label_pad_token_id=dpo.label_pad_token_id,
                    is_encoder_decoder=dpo.is_encoder_decoder,
                ),
                # collate_fn = lambda x: x, # fix weird batching error
                shuffle=False,
                drop_last=False,
            )

        ############################
        # Load classifier model pipeline
        ############################
        else:

            # padding experiments for determinism
            tokenizer.padding_side = "left"
            truncation = False
            if self.args.force_truncation:
                truncation = True
                tokenizer.truncation_side = "left"

            reward_pipeline_kwargs = {
                "batch_size": self.args.batch_size,  # eval_self.args.inference_batch_size,
                "truncation": truncation,
                "padding": True,
                "max_length": self.args.max_length,
                "function_to_apply": "none",  # Compute raw logits
                "return_token_type_ids": False,
            }
            if quantized:
                model_kwargs = {
                    "load_in_8bit": True,
                    "device_map": {"": current_device},
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
                }
            else:
                # note, device map auto does not work for quantized models
                model_kwargs = {"device_map": "auto"}

            model = model_builder(
                self.args.model, **model_kwargs, revision=self.args.revision, trust_remote_code=self.args.trust_remote_code
            )
            reward_pipe = pipeline_builder(
                "text-classification",  # often not used
                model=model,
                tokenizer=tokenizer,
            )

            # set pad token to eos token if not set
            if reward_pipe.tokenizer.pad_token_id is None:
                reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
                reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
            # For models whose config did not contains `pad_token_id`
            if reward_pipe.model.config.pad_token_id is None:
                reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

            # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
            if not check_tokenizer_chat_template(tokenizer):
                reward_pipe.tokenizer.add_eos_token = True

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
            )

            model = accelerator.prepare(reward_pipe.model)
            reward_pipe.model = model

        ############################
        # Run inference
        ############################

        results = []
        scores_chosen = []
        scores_rejected = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if is_dpo:
                rewards_chosen, rewards_rejected = dpo.inference_step(batch)
            else:
                rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen[0], dict):
                score_chosen_batch = [result["score"] for result in rewards_chosen]
                score_rejected_batch = [result["score"] for result in rewards_rejected]
            # for classes that directly output scores (custom code)
            else:
                score_chosen_batch = rewards_chosen.cpu().numpy().tolist()
                score_rejected_batch = rewards_rejected.cpu().numpy().tolist()

            # log results
            [
                results.append(1) if chosen > rejected else results.append(0)
                for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
            ]
            scores_chosen.extend(score_chosen_batch)
            scores_rejected.extend(score_rejected_batch)
        
        accuracy = sum(self.results) / len(self.results)
        final_results = {
            "accuracy": accuracy,
            "num_prompts": len(self.results),
            "model": self.args.model,
            "ref_model": self.args.ref_model,
            "tokenizer": self.args.tokenizer if self.args.tokenizer else self.args.model,
            "chat_template": self.args.chat_template,
        }
        logger.info(f"Final results: {final_results}")

        if self.args.wandb_run is not None:
            for key in final_results:
                wandb_run.summary[f"rewardbench/{key}"] = final_results[key]
            wandb_run.update()
            print(f"Logged metrics to {wandb_run.url}")
        
        self.results = results
        self.scores_chosen = scores_chosen
        self.scores_rejected = scores_rejected
        self.scores = []
        for i in range(len(scores_chosen)):
            self.scores.append(scores_chosen[i])
            self.scores.append(scores_rejected[i])
            
        return self.scores

    def score(self, dataset) -> List[float]:
        all_scores = []
        
        for row in dataset:
            dataset_name, split, max_samples = row["dataset_name"], row["split"], row["max_samples"]
            curr_scores = self._evaluate_reward(dataset_name, split, max_samples)
            all_scores.extend(curr_scores)

        # Chosen, reject interleaving score
        return all_scores
    
    @property
    def min_val(self) -> Optional[float]:
        return None

    @property
    def max_val(self) -> Optional[float]:
        return None

    @property
    def higher_is_better(self) -> bool:
        """Indicates if a higher value is better for this metric."""
        return True
    
    def __eq__(self, other):
        if isinstance(other, RewardBenchModelMetric):
            # Define the sets of attributes to compare
            core_and_inference_args = {
                "dataset", "split", "model", "revision",
                "ref_model", "tokenizer", "chat_template", "not_quantized",
                "batch_size", "max_length"
            }

            # Compare core and inference args
            for attr in core_and_inference_args:
                if getattr(self, attr) != getattr(other, attr):
                    return False
            
            return True
 
        return False
