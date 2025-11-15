"""
Evaluation Results Summarizer for Language Models

This script parses and summarizes evaluation results from the lm-evaluation-harness
framework. It computes average scores across multiple standard NLP benchmarks,
providing a quick overview of model performance.

The script processes results from six key benchmarks that test different aspects
of language understanding and reasoning:
    - ARC Challenge: Science question answering (25-shot)
    - GSM8K: Grade school math problems (5-shot)
    - HellaSwag: Commonsense reasoning (10-shot)
    - MMLU: Multitask language understanding (5-shot)
    - TruthfulQA: Truthfulness in generation (0-shot)
    - WinoGrande: Coreference resolution (5-shot)

These benchmarks are commonly used to evaluate large language models and are part
of standard evaluation suites.

Input Format:
    The script expects a directory containing JSON result files from
    lm-evaluation-harness, with one file per benchmark.

Output:
    Prints individual benchmark scores and an overall average score.

Usage:
    # Basic usage with results directory
    python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M

    # After running evaluations
    python dev/eval/summarize_eval.py path/to/results

Example Output:
    ----------------------------------------
    arc_challenge_25shot.json      : 42.5000
    gsm8k_5shot.json              : 15.3000
    hellaswag_10shot.json         : 76.2000
    mmlu_5shot.json               : 45.8000
    truthfulqa_0shot.json         : 38.1000
    winogrande_5shot.json         : 69.4000
    ----------------------------------------
    Average Score                  : 47.8833

Note:
    This script is optional - the run_eval.sh script should already print these
    statistics. This script is useful for re-analyzing results or generating
    reports after evaluation.

Dependencies:
    - json (standard library)
    - sys (standard library)

Benchmark Details:
    - acc: Accuracy (exact match)
    - acc_norm: Normalized accuracy (for multiple choice with varying options)
    - mc2: Multiple choice score (for TruthfulQA)
"""

import json
import sys

# Get results directory from command-line argument
RESULT = sys.argv[1]
print("-"*40)

# Map each benchmark file to its primary metric
# Different benchmarks use different metrics based on their task type
key = {
    "arc_challenge_25shot.json": "acc_norm",    # Science QA (normalized accuracy)
    "gsm8k_5shot.json": "acc",                  # Math problems (accuracy)
    "hellaswag_10shot.json": "acc_norm",        # Commonsense reasoning (normalized)
    "mmlu_5shot.json": "acc",                   # Multitask understanding (accuracy)
    "truthfulqa_0shot.json": "mc2",             # Truthfulness (multiple choice 2)
    "winogrande_5shot.json": "acc"              # Coreference (accuracy)
}

# Process each benchmark and compute scores
total = 0
for test in ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json",
             "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    # Load benchmark results from JSON file
    data = json.loads(open("./%s/%s" % (RESULT, test)).read())

    # Aggregate scores across all subtasks in this benchmark
    # Some benchmarks have multiple subtasks (e.g., MMLU has many subjects)
    r_count = 0
    r_total = 0
    for test_name in data['results']:
        r_count += 1
        r_total += data['results'][test_name][key[test]]

    # Compute average score for this benchmark (as percentage)
    score = (r_total * 100) / r_count
    print(f"{test:<30} : {score:.4f}")
    total += score

# Compute and print overall average across all 6 benchmarks
average = total / 6.0
print("-"*40)
print(f"Average Score                  : {average:.4f}")
