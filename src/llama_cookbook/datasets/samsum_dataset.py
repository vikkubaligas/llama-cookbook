# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets

from unittest.mock import patch

from huggingface_hub import login
# login(token="")


@patch('builtins.input', return_value="N")
def load_samsum(split, _):
    try:
        ds = datasets.load_dataset("Ankz123/llama3.3_input_output", split=split)
    except ValueError as e:
        if "trust_remote_code" in str(e):
            raise ValueError(
                "Loading Samsung/samsum requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set HF_DATASETS_TRUST_REMOTE_CODE env variable to True.") from e
        else:
            raise e
    return ds


def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = load_samsum(split)



    def apply_prompt_template(sample):
        ds = sample["text"]
        primary = sample["primary"]
        prompt_ip = (
            {
                "role": "system",
                "content": f"""You are an intelligent clinical language model.

        Given a patient's Discharge Summary, Answer clinical questions based on your understanding of the provided medical information.

        Here is the Discharge Summary:

                    {ds}"""
            },
            {"role": "user", "content": "What is the Primary Diagnosis and its respective ICD-10 code?"},
        )

        prompt_op = (
            {"role": "assistant", "content": f"{primary}"},
        )
        return {
            "prompt": tokenizer.apply_chat_template(prompt_ip, tokenize=False, ),
            "summary": tokenizer.apply_chat_template(prompt_op, tokenize=False, )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"], add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
