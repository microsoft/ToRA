"""
This script support LLM API inference with cot/pal/tora prompt.
It can be used to generate tora corpus.
Code based on: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""
import json
import random
import os
import pprint
import re
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from sympy.printing.pretty import pretty

from api.llm_api import llm_api # use your own API like OpenAI API
from utils.python_executor import PythonExecutor
from utils.utils import *
from utils.parser import *
# from utils.trajectory import *
from eval.grader import *
from utils.data_loader import load_data
from infer.inference import prepare_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tora", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_train_prompt_format", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args


def api_with_func_call(engine, prompt, max_tokens, temperature, n, top_p, executor, max_func_call=4, verbose=False):
    if n > 1:
        assert temperature > 0

    if verbose:
        print("\n======= API with function call (START) =======")

    next_batch_queries = [""] * n
    end_queries = []
    for i in range(max_func_call):
        batch_outputs = []
        batch_queries = next_batch_queries
        if len(batch_queries) == 0:
            break
        # get all outputs
        # support batch inference when n > 1
        if i == 0:
            results = llm_api(
                engine=engine, prompt=prompt + batch_queries[0], max_tokens=max_tokens, temperature=temperature,
                n=n, top_p=top_p, stop=["```output\n", "---"],
            )
            batch_outputs.extend(results)
        else:
            for k, query in enumerate(batch_queries):
                print("Call {} / {}".format(k+1, len(batch_queries)))
                results = llm_api(
                    engine=engine, prompt=prompt + query, max_tokens=max_tokens, temperature=temperature,
                    n=1, top_p=top_p, stop=["```output\n", "---"],
                )
                batch_outputs.append(results[0])

        # process all outputs
        next_batch_queries = []
        for query, output in zip(batch_queries, batch_outputs):
            output = output.rstrip()
            query += output
            if verbose:
                print("\n", "-" * 20)
                print(output, end="")
            if "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                prediction, report = executor.apply(program)
                exec_result = prediction if prediction else report
                exec_result = f"\n```output\n{exec_result.strip()}\n```\n"
                query += exec_result
                if verbose:
                    print(exec_result, end="")
                # not end
                if i == max_func_call - 1:
                    query += "\nReach max function call limit."
                next_batch_queries.append(query)
            else:
                end_queries.append(query)

    end_queries.extend(next_batch_queries)
    return end_queries



def main(args):
    examples, processed_samples, out_file = prepare_data(args)
    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    writer = open(out_file, 'w')
    correct, wrong = 0, 0

    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        full_prompt = construct_prompt(args, example)

        # call LLM, return list
        if "tora" in args.prompt_type:
            results = api_with_func_call(
                engine=args.model_name_or_path,
                prompt=full_prompt,
                max_tokens=args.max_tokens_per_call,
                temperature=args.temperature,
                n=args.n_sampling,
                top_p=args.top_p,
                executor=executor,
            )
        else:
            stop_tokens = ["</s>", "---", "```output"]
            if args.prompt_type in ['cot']:
                stop_tokens.append("\n\n")
            results = llm_api(
                engine=args.model_name_or_path,
                prompt=full_prompt,
                max_tokens=args.max_tokens_per_call,
                temperature=args.temperature,
                n=args.n_sampling,
                top_p=args.top_p,
                stop=stop_tokens,
            )
        # deal with error
        if results == ['error']:
            print(">>> Error API call")
            continue
        print("Get {} results".format(len(results)))
        # get prediction
        predictions = []
        reports = []
        for r in results:
            pred, report = run_execute(executor, r, args.prompt_type, execute=True)
            predictions.append(pred)
            reports.append(report)
        print("Executed {} results".format(len(predictions)))
        
        scores = [math_equal(p, gt_ans, timeout=True) for p in predictions]

        is_correct = scores[0]
        if is_correct:
            correct += 1
        else:
            wrong += 1

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans,
            'pred': predictions, 'score': scores}

        if args.prompt_type == "cot":
            sample.update({'code': results})
        elif "tora" in args.prompt_type or "pal" in args.prompt_type:
            sample.update({'report': reports, 'code': results})
        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]

        print(idx)
        show_sample(sample)
        if correct + wrong > 0:
            print("Avg Acc:", correct / (correct + wrong))
        print()
    
        try:
            writer.write(json.dumps(sample) + '\n')
            writer.flush()
        except:
            print(">>> Error writing to file")
            continue

    writer.close()
    print()
    print(correct / (correct + wrong))


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)