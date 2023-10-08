import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print("Saved to", save_path)


def load_prompt(data_name, prompt_type):
    if data_name in ['gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']:
        data_name = "gsm8k"
    if prompt_type in ['platypus_fs', 'wizard_zs']:
        prompt_type = "cot"
    prompt_path = "./prompts/{}/{}.md".format(prompt_type, data_name)
    if not os.path.exists(prompt_path):
        prompt_path = "./prompts/{}.md".format(prompt_type)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as fp:
            prompt = fp.read().strip() + "\n\n"
    else:
        print(f"Error: prompt file {prompt_path} not found")
        prompt = ""
    return prompt

def construct_prompt(args, example):
    demo_prompt = load_prompt(args.data_name, args.prompt_type)
    if args.use_train_prompt_format:
        full_prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
    elif "pot" in args.prompt_type:
        context = f"Question: {example['question']}\n\nSolution:"
        full_prompt = demo_prompt + context
    elif args.prompt_type in ["direct", "cot"]:
        context = f"Question: {example['question']}\nAnswer:"
        full_prompt = demo_prompt + context
    elif args.prompt_type == "pal":
        context = f"Question: {example['question']}"
        full_prompt = demo_prompt + context
    elif args.prompt_type == "wizard_zs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "platypus_fs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt.format(instruction=demo_prompt + f"Question: {example['question']}\nAnswer:")
    else:
        raise NotImplementedError(args.prompt_type)
    return full_prompt

def show_sample(sample):
    print("=="*20)
    print("idx:", sample['idx'])
    for key in ["type", "level"]:
        if key in sample:
            print("{}: {}".format(key, sample[key]))
    print("question:", sample['question'])
    if 'code' in sample:
        for code in sample['code']:
            print('-'*20)
            print("code:", code)
        print("execution", sample['report'])
    for key in ["pred", "gt", "score", "unit", "gt_cot"]:
        if key in sample:
            print("{}: {}".format(key, sample[key]))
    print()