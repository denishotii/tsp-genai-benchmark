# TSP GenAI Benchmark

Comparing hand-coded vs. LLM-generated implementations of classical TSP solvers
(Greedy, Simulated Annealing, Genetic Algorithm) across the TSPLIB benchmark
instances `eil51`, `berlin52`, `ch130`, and `d198`.

Seminar project — Denis Hoti, KU Ingolstadt, SS 2026.
See [CLAUDE.md](CLAUDE.md) for the full project spec.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
src/         reference algorithm implementations
data/tsplib/ TSPLIB benchmark instances (.tsp files)
tests/       unit tests for utils and algorithms
genai_experiments/   prompts and LLM-generated code
results/     benchmark CSVs
notebooks/   analysis & visualization
paper/       final write-up
```
