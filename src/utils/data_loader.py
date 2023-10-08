import os
import json
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from utils.utils import load_jsonl

def load_data(data_name, split):
    data_file = f"data/{data_name}/{split}.json"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset("competition_math", split=split, name="main", cache_dir="data_name/temp")
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split, name="main")
        elif data_name == "gsm-hard":
            dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        elif data_name == "svamp":
            # evaluate on training set + test set 
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets([dataset, load_dataset("ChilleD/SVAMP", split="test")])
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(lambda x: ";" not in x['answer']) # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"data_name/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example['type'] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "finqa":
            dataset = load_dataset("dreamerdeo/finqa", split=split, name="main")
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "tabmwp":
            examples = []
            with open(f"data_name/tabmwp/tabmwp_{split}.json", "r") as f:
                data_dict = json.load(f)
                examples.extend(data_dict.values())
            dataset = Dataset.from_list(examples)
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "bbh":
            examples = []
            for data_name in ["reasoning_about_colored_objects", "penguins_in_a_table",\
                            "date_understanding", "repeat_copy_logic", "object_counting"]:
                with open(f"data_name/bbh/bbh/{data_name}.json", "r") as f:
                    sub_examples = json.load(f)["examples"]
                    for example in sub_examples:
                        example['type'] = data_name
                    examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        else:
            raise NotImplementedError(data_name)

        if 'idx' not in dataset.column_names:
            dataset = dataset.map(lambda x, i: {'idx': i, **x}, with_indices=True)

        os.makedirs(f"data_name/{data_name}", exist_ok=True)
        dataset.to_json(data_file)
        examples = list(dataset)

    # dedepulicate & sort
    examples = {example['idx']: example for example in examples}
    examples = list(examples.values())
    examples = sorted(examples, key=lambda x: x['idx'])
    return examples