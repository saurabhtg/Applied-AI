# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

#!/usr/bin/env bash
#
# ci/run_evals.sh
#
# Run evals against every prompt in prompts/.
# Exit 0 if all pass, non-zero if any fail.
# Called from GitHub Actions on every PR.
#
# If you're seeing "STATUS: FAIL" here, it means your prompt change
# regressed eval scores. The diff is in the scores.json file next to
# the prompt. Read it. Fix it. Don't merge until it's green.

set -euo pipefail

PROMPTS_DIR="${1:-vol1_foundations/part04_prompts_as_code/prompts}"
EXIT_CODE=0
FAILED_PROMPTS=()

if [[ ! -d "$PROMPTS_DIR" ]]; then
    echo "Error: prompts directory not found at $PROMPTS_DIR"
    exit 2
fi

echo "================================================"
echo "Running evals against all prompts in $PROMPTS_DIR"
echo "================================================"
echo

for prompt_dir in "$PROMPTS_DIR"/*/; do
    prompt_name=$(basename "$prompt_dir")

    if [[ ! -f "${prompt_dir}prompt.txt" ]]; then
        echo "Skip: $prompt_name (no prompt.txt)"
        continue
    fi
    if [[ ! -f "${prompt_dir}eval.jsonl" ]]; then
        echo "Skip: $prompt_name (no eval.jsonl)"
        continue
    fi

    echo "--- $prompt_name ---"
    if python -m vol1_foundations.part04_prompts_as_code.eval_runner "$prompt_dir"; then
        echo "✓ $prompt_name passed"
    else
        echo "✗ $prompt_name FAILED"
        EXIT_CODE=1
        FAILED_PROMPTS+=("$prompt_name")
    fi
    echo
done

echo "================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "All eval sets passed."
else
    echo "Eval failures in: ${FAILED_PROMPTS[*]}"
    echo "Fix the prompts or update the eval cases before merging."
fi
echo "================================================"

exit $EXIT_CODE
