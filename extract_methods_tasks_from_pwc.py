'''
python extract_methods_tasks_from_pwc.py \
    --evaluation-tables-file evaluation-tables.json \
    --methods-file methods.json  \
    --output-methods-list methods.csv \
    --output-tasks-list tasks.csv
'''

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation-tables-file', type=str, required=True, help="Downloaded from /projects/metis0_ssd/users/vijayv/dataset-recommendation/")
parser.add_argument('--methods-file', type=str, required=True, help="Downloaded from /projects/metis0_ssd/users/vijayv/dataset-recommendation/")
parser.add_argument('--output-methods-list-file', type=str, default="methods.csv")
parser.add_argument('--output-tasks-list-file', type=str, default="tasks.csv")

def parse_tasks_from_evaluation_tables_file(evaluation_tables_file):
    tables = json.load(open(evaluation_tables_file))
    tasks = [row["task"] for row in tables if len(row["task"]) > 0]
    return tasks

def parse_methods_from_methods_file(methods_file):
    methods = json.load(open(methods_file))
    names = [row["name"] for row in methods if len(row["name"]) > 0]
    return names

def isSubArray(A, B):
    'taken from https://www.geeksforgeeks.org/check-whether-an-array-is-subarray-of-another-array/'

    # Two pointers to traverse the arrays
    i = 0; j = 0;
 
    # Traverse both arrays simultaneously
    while (i < len(A) and j < len(B)):
 
        # If element matches
        # increment both pointers
        if (A[i] == B[j]):
 
            i += 1;
            j += 1;
 
            # If array B is completely
            # traversed
            if (j == len(B)):
                return True;
         
        # If not,
        # increment i and reset j
        else:
            i = i - j + 1;
            j = 0;
         
    return False;

def construct_prompt(items, prefix):
    if len(items) == 0:
        prompt = f"No {prefix} specified"
    elif len(items) == 1:
        prompt = f"The {prefix} is {items[0]}"
    elif len(items) == 2:
        prompt = f"The {prefix}s are {items[0]} and {items[1]}"
    else:
        prompt = f"{prefix}s are "
        for i in range(len(items)):
            if i < len(items) - 2:
                prompt = prompt + items[i] + ", "
            elif i == len(items) - 2:
                prompt = prompt + items[i] + ", and "
            elif i == len(items) - 1:
                prompt = prompt + items[i]
    return prompt

def add_prompt_to_description(description, tasks, methods):
    description_tokens = description.lower().split()
    matching_tasks = []
    for t in tasks:
        task_tokens = t.lower().split()
        if isSubArray(description_tokens, task_tokens):
            matching_tasks.append(t)
    matching_tasks = sorted(matching_tasks)
    
    matching_methods = []
    for m in methods:
        method_tokens = m.lower().split()
        if isSubArray(description_tokens, method_tokens):
            matching_methods.append(m)
    matching_methods = sorted(matching_methods)
    
    tasks_prefix = construct_prompt(matching_tasks, "task")
    methods_prefix = construct_prompt(matching_methods, "method")
    description = f"{tasks_prefix} [SEP] {methods_prefix} [SEP] {description}"
    return description

if __name__ == "__main__":
    args = parser.parse_args()
    tasks = parse_tasks_from_evaluation_tables_file(args.evaluation_tables_file)
    with open(args.output_tasks_list_file, 'w') as f:
        f.write("\n".join(tasks))
    methods = parse_methods_from_methods_file(args.methods_file)
    with open(args.output_methods_list_file, 'w') as f:
        f.write("\n".join(methods))
    
    description = "I want to build a video recommendation system using convolutional neural networks"
    add_prompt_to_description(description, tasks, methods)