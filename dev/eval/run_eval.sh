#!/bin/bash
# ==============================================================================
# Open LLM Leaderboard Evaluation Script
# ==============================================================================
# This script evaluates language models using the standardized benchmarks from
# the Open LLM Leaderboard (HuggingFace). It runs six comprehensive evaluations
# to measure model performance across different capabilities.
#
# Reference:
#   https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
#   See "About" tab -> "REPRODUCIBILITY" for methodology details
#
# Evaluation benchmarks:
#   1. TruthfulQA (0-shot)   - Measures truthfulness and factual accuracy
#   2. WinoGrande (5-shot)   - Tests commonsense reasoning
#   3. ARC Challenge (25-shot) - Science question answering
#   4. HellaSwag (10-shot)   - Commonsense natural language inference
#   5. GSM8K (5-shot)        - Grade school math problems
#   6. MMLU (5-shot)         - Massive multitask language understanding
#
# Prerequisites:
#   1. Clone the evaluation harness:
#      git clone https://github.com/EleutherAI/lm-evaluation-harness/
#      cd lm-evaluation-harness
#      git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
#      pip install -e .
#      cd ..
#
#   2. Ensure CUDA is available for GPU evaluation
#
# Usage:
#   ./dev/eval/run_eval.sh [model_path] [result_name]
#
# Arguments:
#   model_path   - HuggingFace model (e.g., "openai-community/gpt2")
#                  OR local path (e.g., "./gpt2-124M-run1")
#   result_name  - Output folder name for results
#
# Examples:
#   ./dev/eval/run_eval.sh openai-community/gpt2 gpt2_baseline
#   ./dev/eval/run_eval.sh ./log_gpt2_124M my_trained_model
#
# Running in background (recommended for long evaluations):
#   nohup ./dev/eval/run_eval.sh [model_path] [result_name] > run.txt 2> err.txt &
#
# Runtime:
#   - Small models (124M): ~1-2 hours
#   - Medium models (350M): ~2-4 hours
#   - Large models (1.5B+): ~4-8 hours
#
# Output:
#   - Results saved to: lm-evaluation-harness/results/[result_name]/
#   - Summary generated at the end
# ==============================================================================

# ==============================================================================
# Argument Validation
# ==============================================================================

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Error: Missing model path argument"
    echo ""
    echo "Usage: ./run_eval.sh [model_path] [result_name]"
    echo "Example: ./run_eval.sh openai-community/gpt2 my_result"
    exit 1
fi

# Check if result name is provided
if [ -z "$2" ]; then
    echo "Error: Missing result name argument"
    echo ""
    echo "Usage: ./run_eval.sh [model_path] [result_name]"
    echo "Example: ./run_eval.sh openai-community/gpt2 my_result"
    exit 1
fi

# ==============================================================================
# Configuration
# ==============================================================================

# Convert model path to absolute path (handles both HF models and local paths)
export MODEL="$(realpath -s "$1")"
export RESULT="$2"

echo "=========================================="
echo "Open LLM Leaderboard Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Results directory: ./lm-evaluation-harness/results/$RESULT"
echo ""
echo "This will run 6 benchmark evaluations:"
echo "  1. TruthfulQA (0-shot)"
echo "  2. WinoGrande (5-shot)"
echo "  3. ARC Challenge (25-shot)"
echo "  4. HellaSwag (10-shot)"
echo "  5. GSM8K (5-shot)"
echo "  6. MMLU (5-shot)"
echo ""
echo "Estimated runtime: 1-8 hours depending on model size"
echo "=========================================="
echo ""

# Change to evaluation harness directory
cd lm-evaluation-harness

# ==============================================================================
# Benchmark Evaluations
# ==============================================================================

# Common arguments for all evaluations
# --model hf-causal-experimental: Use HuggingFace causal LM loader
# --model_args: Model configuration
#   - pretrained=$MODEL: Path to model
#   - use_accelerate=True: Use HF Accelerate for efficiency
#   - trust_remote_code=True: Allow custom model code
# --batch_size 1: Process one example at a time (safer for memory)
# --no_cache: Don't use cached results
# --write_out: Save detailed outputs
# --device cuda: Use GPU acceleration

echo "[1/6] Running TruthfulQA (0-shot)..."
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True \
    --tasks truthfulqa_mc \
    --batch_size 1 \
    --no_cache \
    --write_out \
    --output_path results/$RESULT/truthfulqa_0shot.json \
    --device cuda

echo "[2/6] Running WinoGrande (5-shot)..."
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True \
    --tasks winogrande \
    --batch_size 1 \
    --no_cache \
    --write_out \
    --output_path results/$RESULT/winogrande_5shot.json \
    --device cuda \
    --num_fewshot 5

echo "[3/6] Running ARC Challenge (25-shot)..."
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True \
    --tasks arc_challenge \
    --batch_size 1 \
    --no_cache \
    --write_out \
    --output_path results/$RESULT/arc_challenge_25shot.json \
    --device cuda \
    --num_fewshot 25

echo "[4/6] Running HellaSwag (10-shot)..."
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True \
    --tasks hellaswag \
    --batch_size 1 \
    --no_cache \
    --write_out \
    --output_path results/$RESULT/hellaswag_10shot.json \
    --device cuda \
    --num_fewshot 10

echo "[5/6] Running GSM8K (5-shot)..."
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True \
    --tasks gsm8k \
    --batch_size 1 \
    --no_cache \
    --write_out \
    --output_path results/$RESULT/gsm8k_5shot.json \
    --device cuda \
    --num_fewshot 5

echo "[6/6] Running MMLU (5-shot) - 57 diverse tasks..."
# MMLU (Massive Multitask Language Understanding) covers 57 subjects
# Including STEM, humanities, social sciences, and more
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,trust_remote_code=True \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --batch_size 1 \
    --no_cache \
    --write_out \
    --output_path results/$RESULT/mmlu_5shot.json \
    --device cuda \
    --num_fewshot 5

# ==============================================================================
# Results Summary
# ==============================================================================

# Return to root directory
cd ..

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo "Generating summary..."
echo ""

# Run summary script to aggregate results
python dev/eval/summarize_eval.py lm-evaluation-harness/results/$RESULT

echo ""
echo "Results saved to: lm-evaluation-harness/results/$RESULT"
echo "=========================================="
