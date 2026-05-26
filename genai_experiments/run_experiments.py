"""Run the GenAI experiment matrix and log results.

For each (model x strategy x algorithm) cell: build the prompt, generate code
via the LLM, extract it, save the artifacts (prompt + raw response + code),
run it on every TSPLIB instance, and append per-instance rows to
``experiment_log.csv``.

The iterative-refinement strategy runs a generate -> execute -> feed-back loop
(up to ``--max-rounds``) until the code executes successfully on a fast
reference instance, then the final code is benchmarked like any other.

Run from the project root (needs API keys in .env, unless --dry-run):

    python -m genai_experiments.run_experiments                # full matrix
    python -m genai_experiments.run_experiments --dry-run       # offline test
    python -m genai_experiments.run_experiments --models claude --strategies zero_shot --algorithms greedy
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from genai_experiments.code_extraction import extract_code
from genai_experiments.executor import ExecutionResult, run_generated_solver
from genai_experiments.llm_clients import MODELS, LLMResponse, generate
from genai_experiments.prompt_builder import (
    ALGORITHMS,
    STRATEGIES,
    build_feedback_prompt,
    build_prompt,
)
from src.utils.distance import distance_matrix
from src.utils.parser import parse_tsp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "tsplib"
GENERATED_CODE_DIR = Path(__file__).resolve().parent / "generated_code"
LOG_PATH = Path(__file__).resolve().parent / "experiment_log.csv"

TSPLIB_OPTIMA = {"eil51": 426, "berlin52": 7542, "ch130": 6110, "d198": 15780}

# Strategy name -> on-disk folder (CLAUDE.md §6 uses short folder names).
STRATEGY_DIR = {
    "zero_shot": "zero_shot",
    "few_shot": "few_shot",
    "chain_of_thought": "cot",
    "iterative_refinement": "iterative",
}

# Fast instance used to test code during iterative-refinement rounds.
REFINEMENT_INSTANCE = "eil51"

# Canned valid solver used by --dry-run so the full pipeline runs offline.
_DRY_RUN_RESPONSE = """Here is the implementation:

```python
import numpy as np

def solve(dist_matrix):
    n = dist_matrix.shape[0]
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        cur = tour[-1]
        best, best_d = -1, float("inf")
        for j in range(n):
            if not visited[j] and dist_matrix[cur][j] < best_d:
                best_d, best = dist_matrix[cur][j], j
        tour.append(best)
        visited[best] = True
    length = sum(dist_matrix[tour[k]][tour[(k + 1) % n]] for k in range(n))
    return tour, length
```
"""


@dataclass
class GenerationResult:
    """Outcome of generating one solver (possibly across refinement rounds)."""

    code: str | None
    prompt: str           # full prompt transcript (all rounds, for iterative)
    response: str         # full raw response transcript
    input_tokens: int | None
    output_tokens: int | None
    rounds: int
    error: str | None = None   # set if the API call itself failed


def _call_model(model_key: str, messages: list[dict], dry_run: bool) -> LLMResponse:
    if dry_run:
        return LLMResponse(text=_DRY_RUN_RESPONSE, model=f"{model_key}-dry-run")
    return generate(model_key, messages)


def _format_feedback(result: ExecutionResult) -> str:
    """Turn an execution failure into natural-language feedback for the model."""
    if result.status == "timeout":
        return f"The code did not finish within the time limit ({result.error})."
    if result.status == "invalid_tour":
        return f"The code ran but returned an invalid tour: {result.error}"
    if result.status == "no_solve":
        return "The code did not define a function named `solve`."
    if result.status == "bad_output":
        return f"The solve function returned the wrong type: {result.error}"
    return f"The code raised an error:\n\n{result.error}"


def _generate_single(model_key: str, strategy: str, algorithm: str, dry_run: bool) -> GenerationResult:
    prompt = build_prompt(strategy, algorithm)
    try:
        response = _call_model(model_key, [{"role": "user", "content": prompt}], dry_run)
    except Exception as exc:  # API/network failure — record and move on
        return GenerationResult(None, prompt, "", None, None, rounds=1, error=str(exc))
    return GenerationResult(
        code=extract_code(response.text),
        prompt=prompt,
        response=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        rounds=1,
    )


def _generate_iterative(
    model_key: str,
    algorithm: str,
    ref_matrix,
    timeout: float,
    max_rounds: int,
    dry_run: bool,
) -> GenerationResult:
    seed = build_prompt("iterative_refinement", algorithm)
    messages: list[dict] = [{"role": "user", "content": seed}]
    prompt_log = [f"=== round 1 prompt ===\n{seed}"]
    response_log: list[str] = []
    in_tokens = out_tokens = 0
    code = None

    for rnd in range(1, max_rounds + 1):
        try:
            response = _call_model(model_key, messages, dry_run)
        except Exception as exc:
            return GenerationResult(
                code, "\n\n".join(prompt_log), "\n\n".join(response_log),
                in_tokens or None, out_tokens or None, rounds=rnd, error=str(exc),
            )
        in_tokens += response.input_tokens or 0
        out_tokens += response.output_tokens or 0
        response_log.append(f"=== round {rnd} response ===\n{response.text}")
        code = extract_code(response.text)

        result = run_generated_solver(code, ref_matrix, timeout=timeout)
        if result.ok or rnd == max_rounds:
            return GenerationResult(
                code, "\n\n".join(prompt_log), "\n\n".join(response_log),
                in_tokens or None, out_tokens or None, rounds=rnd,
            )

        feedback = build_feedback_prompt(_format_feedback(result))
        messages.append({"role": "assistant", "content": response.text})
        messages.append({"role": "user", "content": feedback})
        prompt_log.append(f"=== round {rnd + 1} prompt ===\n{feedback}")

    return GenerationResult(
        code, "\n\n".join(prompt_log), "\n\n".join(response_log),
        in_tokens or None, out_tokens or None, rounds=max_rounds,
    )


def _save_artifacts(model_key: str, strategy: str, algorithm: str, gen: GenerationResult) -> None:
    out_dir = GENERATED_CODE_DIR / model_key / STRATEGY_DIR[strategy]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{algorithm}.prompt.txt").write_text(gen.prompt)
    (out_dir / f"{algorithm}.response.txt").write_text(gen.response)
    if gen.code is not None:
        (out_dir / f"{algorithm}.py").write_text(gen.code)


def run(
    models: list[str],
    strategies: list[str],
    algorithms: list[str],
    timeout: float,
    max_rounds: int,
    dry_run: bool,
) -> pd.DataFrame:
    instances = {
        name: (len(coords := parse_tsp(DATA_DIR / f"{name}.tsp")), distance_matrix(coords))
        for name in TSPLIB_OPTIMA
    }
    ref_matrix = instances[REFINEMENT_INSTANCE][1]

    rows = []
    for model_key in models:
        for strategy in strategies:
            for algorithm in algorithms:
                label = f"{model_key}/{strategy}/{algorithm}"
                print(f"\n>>> generating {label}")
                if strategy == "iterative_refinement":
                    gen = _generate_iterative(
                        model_key, algorithm, ref_matrix, timeout, max_rounds, dry_run
                    )
                else:
                    gen = _generate_single(model_key, strategy, algorithm, dry_run)
                _save_artifacts(model_key, strategy, algorithm, gen)

                if gen.error is not None:
                    print(f"    generation FAILED: {gen.error}")
                if gen.code is None:
                    for instance, optimum in TSPLIB_OPTIMA.items():
                        rows.append(_row(model_key, strategy, algorithm, instance,
                                         instances[instance][0], optimum, gen, None))
                    continue

                for instance, optimum in TSPLIB_OPTIMA.items():
                    n, matrix = instances[instance]
                    result = run_generated_solver(gen.code, matrix, timeout=timeout)
                    rows.append(_row(model_key, strategy, algorithm, instance, n, optimum, gen, result))
                    gap = (f"{(result.length - optimum) / optimum * 100:.2f}%"
                           if result.length is not None else "-")
                    print(f"    {instance:<10} {result.status:<13} gap={gap}")

    return pd.DataFrame(rows)


def _row(model_key, strategy, algorithm, instance, cities, optimum, gen, result) -> dict:
    if result is None:
        status, length, gap, runtime, reported = "generation_failed", None, None, None, None
    else:
        status = result.status
        length = int(result.length) if result.length is not None else None
        gap = round((result.length - optimum) / optimum * 100, 2) if result.length is not None else None
        runtime = round(result.runtime_s, 3) if result.runtime_s is not None else None
        reported = result.reported_length
    return {
        "model": model_key,
        "strategy": strategy,
        "algorithm": algorithm,
        "instance": instance,
        "cities": cities,
        "status": status,
        "ran": status == "ok",
        "length": length,
        "optimum": optimum,
        "gap_pct": gap,
        "runtime_s": runtime,
        "reported_length": reported,
        "refinement_rounds": gen.rounds,
        "input_tokens": gen.input_tokens,
        "output_tokens": gen.output_tokens,
        "generation_error": gen.error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TSP GenAI experiment matrix.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--strategies", nargs="+", choices=list(STRATEGIES), default=list(STRATEGIES))
    parser.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS), default=list(ALGORITHMS))
    parser.add_argument("--timeout", type=float, default=60.0, help="per-run wall-clock limit (s)")
    parser.add_argument("--max-rounds", type=int, default=3, help="iterative-refinement rounds")
    parser.add_argument("--dry-run", action="store_true", help="use a canned response; no API calls")
    args = parser.parse_args()

    df = run(args.models, args.strategies, args.algorithms,
             args.timeout, args.max_rounds, args.dry_run)

    # Append to the log if it exists (so partial runs accumulate), else create.
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

    print(f"\nWrote {len(df)} total rows to {LOG_PATH.relative_to(PROJECT_ROOT)}")
    ran = df["ran"].sum() if "ran" in df else 0
    print(f"Rows where generated code ran successfully: {ran} / {len(df)}")


if __name__ == "__main__":
    main()
