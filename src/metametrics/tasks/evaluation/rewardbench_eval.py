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

import os
import numpy as np
import json

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

def evaluate_rewardbench(scores, output_path):
    from rewardbench import load_eval_dataset
    from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
    from rewardbench.utils import calculate_scores_per_section

    # primary set compiles slightly more information
    dataset, subsets = load_eval_dataset(
        core_set=True,
        conv=None, # We don't need conv, we only care about the subsets
        custom_dialogue_formatting=False,
        tokenizer=None, # We don't need tokenizer, we only care about the subsets
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "prompt"],
    )
    
    scores_chosen = scores[::2]
    scores_rejected = scores[1::2]
    results = [1 if chosen > rejected else 0 for chosen, rejected in zip(scores_chosen, scores_rejected)]

    # calculate accuracy
    accuracy = sum(results) / len(results)
    logger.info(f"Results: {accuracy}, on {len(results)} prompts")

    # compute mean and std of scores, chosen and rejected, then margin between them
    logger.info(f"Mean chosen: {np.mean(scores_chosen)}, std: {np.std(scores_chosen)}")
    logger.info(f"Mean rejected: {np.mean(scores_rejected)}, std: {np.std(scores_rejected)}")
    logger.info(f"Mean margin: {np.mean(np.array(scores_chosen) - np.array(scores_rejected))}")

    out_dataset = dataset.add_column("results", results)
    out_dataset = out_dataset.add_column("subsets", subsets)
    out_dataset = out_dataset.to_pandas()  # I know this is meh

    results_grouped = {}
    present_subsets = np.unique(out_dataset["subsets"])
    for subset in present_subsets:
        subset_dataset = out_dataset[out_dataset["subsets"] == subset]
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        logger.info(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    results_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
    logger.info(f"Results: {results_section}")
    
    with open(os.path.join(output_path, "rewardbench_result.json"), "w") as f:
        json.dump(results_section, f, indent=4)
