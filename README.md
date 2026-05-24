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

## TSPLIB data provenance

The four benchmark instances in `data/tsplib/` (`eil51`, `berlin52`, `ch130`,
`d198`) were downloaded on 2026-05-24 from the **Rice University softlib
mirror** of TSPLIB95:

```
http://softlib.rice.edu/pub/tsplib/tsp/<name>.tsp.gz
```

The canonical TSPLIB95 distribution is hosted by Reinelt at Heidelberg
(`comopt.ifi.uni-heidelberg.de/software/TSPLIB95/`), but that host was
unreachable at download time. Rice's softlib is a long-established academic
mirror of the same files; content was byte-verified against an independent
mirror before commit. All four files use `EDGE_WEIGHT_TYPE: EUC_2D`.

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
