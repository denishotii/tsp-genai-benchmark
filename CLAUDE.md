# CLAUDE.md — Project Context & Working Guide

> This file gives Claude Code full context on what this project is, why it exists, what needs to be built, and how to build it. Read this before doing any work.

---

## 1. Who and Where

- **Student:** Denis Hoti
- **University:** KU Ingolstadt (Katholische Universität Eichstätt-Ingolstadt)
- **Semester:** Summer Semester 2026 (SS 2026)
- **Course:** Digital Seminar in Data Science & Quantitative Applications
- **Paired with:** Pro-Seminar "Generative AI in Logistics" (SS 2026)
- **Topic assignment:** P3 (Traveling Salesman Problem) + L2 (Metaheuristic Methods: SA + GA)
- **Working solo** — no group partner

---

## 2. What This Project Is

A research project that answers one central question:

> **Can Generative AI (LLMs) reliably implement complex combinatorial optimization algorithms — and which model does it better?**

The Traveling Salesman Problem (TSP) is used as the test bed. We implement TSP solvers two ways:

1. **By hand** (reference implementations, following academic literature)
2. **Via LLM prompting** (asking GPT and Claude to generate the same code)

Then we benchmark both on the same standard instances and measure the gap.

---

## 3. The Problem — TSP

The **Traveling Salesman Problem** asks:
> Given a set of cities and distances between them, find the shortest route that visits each city exactly once and returns to the start.

- Formally: find the minimum-cost Hamiltonian cycle in a complete weighted graph
- **NP-hard** — no polynomial-time exact solution known
- With 20 cities: 2.4 quintillion (20!/2) possible tours — brute force is impossible
- Real-world applications: delivery routing, warehouse picking, service scheduling, PCB drilling

**Benchmark instances** — from TSPLIB (Reinelt, 1991):
| Instance | Cities | Known Optimal |
|----------|--------|---------------|
| eil51    | 51     | 426           |
| berlin52 | 52     | 7542          |
| ch130    | 130    | 6110          |
| d198     | 198    | 15780         |

All instances use **Euclidean 2D distances** (EUC_2D in TSPLIB format).

---

## 4. Three Solution Methods (Track A — Reference Implementations)

These are implemented **by hand in Python**, following academic literature. They serve as the quality baseline.

### 4.1 Greedy Nearest-Neighbor (Baseline)
- Constructive heuristic: always go to the nearest unvisited city
- Fast, simple, produces a valid tour but suboptimal (~20–25% above optimal)
- Purpose: lower bound on quality — any serious solver should beat this

### 4.2 Simulated Annealing (SA)
- Based on: Kirkpatrick, Gelatt & Vecchi (1983)
- **Neighborhood:** 2-opt (reverse a segment of the tour)
- **Cooling schedule:** geometric — `T = T * alpha`, with `alpha ≈ 0.995`
- **Starting temperature:** calibrated so ~80% of moves accepted initially
- **Stopping criterion:** temperature below threshold or no improvement for N iterations
- Key parameters: `T_initial`, `T_final`, `alpha`, `max_iterations`

### 4.3 Genetic Algorithm (GA)
- **Representation:** permutation encoding (list of city indices)
- **Selection:** tournament selection (k=5)
- **Crossover:** Order Crossover (OX) — preserves relative order of cities
- **Mutation:** swap mutation (swap two random cities in the tour)
- **Population size:** ~100–200 individuals
- **Generations:** ~500–1000
- **Elitism:** top 1–2 individuals always survive

---

## 5. GenAI Evaluation (Track B — LLM Experiments)

### 5.1 Models to Compare
- **GPT-5.5** (OpenAI) — via API
- **Claude Opus 4.7** (Anthropic, `claude-opus-4-7`) — via API
- Both models are asked to implement the exact same algorithms

> **Note (2026-05):** The originally specced models (GPT-4o / GPT-4.1 and
> Claude Sonnet 4.6) were superseded — GPT-4o/4.1 were retired from the
> OpenAI API around April 2026. We updated to a current **flagship-vs-
> flagship** pairing (GPT-5.5 vs Claude Opus 4.7) for a fair, tier-matched
> comparison. Record this substitution in the paper's methodology section.

### 5.2 Prompting Strategies (4 total)
For each algorithm × each model, we test all four strategies:

| Strategy | Description |
|----------|-------------|
| **Zero-shot** | "Implement a Simulated Annealing solver for TSP in Python." — no examples |
| **Few-shot** | Provide 1–2 examples of related optimization code before the ask |
| **Chain-of-thought (CoT)** | Ask the model to reason step-by-step before coding |
| **Iterative refinement** | Start with zero-shot, then provide feedback/error messages and re-prompt |

### 5.3 Evaluation Matrix
Every experiment is a cell in this matrix:

```
Prompting Strategy (4) × Model (2) × Algorithm (3) × TSPLIB Instance (4)
= up to 96 data points
```

### 5.4 What We Measure
- **Correctness:** Does the code run without errors? Does it produce a valid tour?
- **Solution quality:** Gap-to-optimal (%) = `(found_tour - optimal) / optimal * 100`
- **Code quality (qualitative):** Readability, structure, adherence to algorithm spec

### 5.5 Prompting Log
Every prompt sent to every model must be logged with:
- Model name + version
- Strategy used
- Full prompt text
- Full response (code)
- Whether code ran without modification
- Tour length achieved on each TSPLIB instance
- Gap-to-optimal

---

## 6. Project Structure

```
tsp-genai-benchmark/
├── CLAUDE.md                  # This file
├── LICENSE                    # MIT
├── README.md                  # Public-facing project overview
│
├── data/
│   └── tsplib/                # TSPLIB .tsp files (eil51, berlin52, ch130, d198)
│
├── src/
│   ├── utils/
│   │   ├── parser.py          # Parse .tsp files → list of (x, y) coordinates
│   │   ├── distance.py        # Euclidean distance matrix computation
│   │   └── tour.py            # Tour validation, tour length calculation
│   │
│   ├── algorithms/
│   │   ├── greedy.py          # Nearest-neighbor greedy heuristic
│   │   ├── simulated_annealing.py   # SA with 2-opt, geometric cooling
│   │   └── genetic_algorithm.py     # GA with OX crossover, tournament selection
│   │
│   └── benchmark/
│       └── run_benchmark.py   # Run all algorithms on all instances, output results
│
├── genai_experiments/
│   ├── prompts/               # Prompt templates for each strategy
│   │   ├── zero_shot.txt
│   │   ├── few_shot.txt
│   │   ├── chain_of_thought.txt
│   │   └── iterative_refinement.txt
│   │
│   ├── generated_code/        # Raw code output from each LLM experiment
│   │   ├── gpt/
│   │   │   ├── zero_shot/
│   │   │   ├── few_shot/
│   │   │   ├── cot/
│   │   │   └── iterative/
│   │   └── claude/
│   │       ├── zero_shot/
│   │       ├── few_shot/
│   │       ├── cot/
│   │       └── iterative/
│   │
│   └── experiment_log.csv     # Full log of all experiments (see §5.5)
│
├── results/
│   ├── benchmark_results.csv  # Reference implementation results
│   ├── genai_results.csv      # LLM-generated code results
│   └── comparison_table.csv   # Side-by-side gap-to-optimal comparison
│
├── notebooks/
│   └── analysis.ipynb         # Visualization and analysis of results
│
└── paper/                     # Academic paper (LaTeX or Word)
    └── TSP_GenAI_Denis_Hoti.docx
```

---

## 7. Implementation Priority & Order

Work in this order — each step depends on the previous:

### Phase 1 — Foundation (late May)
- [ ] Download TSPLIB instances: eil51.tsp, berlin52.tsp, ch130.tsp, d198.tsp
- [ ] `src/utils/parser.py` — parse `.tsp` files into coordinate lists
- [ ] `src/utils/distance.py` — compute Euclidean distance matrix
- [ ] `src/utils/tour.py` — `tour_length(tour, dist_matrix)` and `is_valid_tour(tour, n)`
- [ ] Write unit tests: verify eil51 has 51 nodes, distances are symmetric, etc.

### Phase 2 — Reference Algorithms (May–early June)
- [ ] `src/algorithms/greedy.py` — nearest-neighbor baseline
- [ ] `src/algorithms/simulated_annealing.py` — SA with 2-opt + geometric cooling
- [ ] `src/algorithms/genetic_algorithm.py` — GA with OX + tournament selection
- [ ] `src/benchmark/run_benchmark.py` — run all 3 × 4 = 12 combinations, log results
- [ ] Verify results are reasonable (within ~10–15% of known optimal for SA/GA)

### Phase 3 — GenAI Experiments (mid June)
- [ ] Write prompt templates for all 4 strategies
- [ ] Run zero-shot prompts on both GPT and Claude for all 3 algorithms
- [ ] Run few-shot, CoT, iterative prompts
- [ ] Save all generated code to `genai_experiments/generated_code/`
- [ ] Benchmark every piece of generated code on all 4 TSPLIB instances
- [ ] Fill `experiment_log.csv`

### Phase 4 — Analysis (late June)
- [ ] Build `results/comparison_table.csv`
- [ ] `notebooks/analysis.ipynb` — plots and tables for the paper
- [ ] Key figures needed: gap-to-optimal bar chart, prompting strategy comparison heatmap, model comparison table

### Phase 5 — Paper (July)
- [ ] Write 8-section paper (see §8)
- [ ] Finalize visualizations
- [ ] Prepare final presentation

---

## 8. Paper Structure (8 Sections)

1. **Introduction** — motivation, research questions
2. **TSP Formalization** — mathematical definition, complexity
3. **Real-World Context** — last-mile delivery routing application
4. **Solution Methods** — Greedy, SA, GA (theory + our implementation choices)
5. **GenAI Experiment Design** — models, prompting strategies, methodology
6. **Results & Benchmarks** — tables, charts, gap-to-optimal analysis
7. **Critical Reflection** — where AI succeeded, where it failed, failure modes by strategy
8. **Conclusion & Future Work**

---

## 9. Coding Standards

- **Language:** Python 3.10+
- **Style:** PEP 8, type hints where practical
- **No external optimization libraries** (no OR-Tools, no scipy.optimize) — algorithms must be implemented from scratch
- **Reproducibility:** set random seeds (`random.seed(42)`, `numpy.random.seed(42)`) everywhere
- **Dependencies:** numpy, pandas, matplotlib — keep it minimal
- **Every algorithm function must accept:** `dist_matrix` (2D numpy array) and return `(best_tour: list[int], best_length: float)`

---

## 10. Literature Papers

All papers are stored in `literature papers/REFERENCES.md`, which includes direct PDF download links, APA citations, and notes on which sections to read. Here is what each paper is for and how it connects to the code:

### Reinelt (1991) — TSPLIB
- **File:** `literature papers/Reinelt_1991_TSPLIB.pdf` *(download via KU library — see REFERENCES.md)*
- **Used for:** Understanding the `.tsp` file format parsed in `src/utils/parser.py`, and the known optimal tour lengths used in gap calculations throughout `src/benchmark/run_benchmark.py`
- **Cite when:** Describing the benchmark instances (eil51, berlin52, ch130, d198) in the paper

### Kirkpatrick, Gelatt & Vecchi (1983) — Simulated Annealing
- **File:** `literature papers/Kirkpatrick_1983_SimulatedAnnealing.pdf` *(direct PDF in REFERENCES.md)*
- **Used for:** The theoretical basis of `src/algorithms/simulated_annealing.py` — the Metropolis criterion (accept/reject logic), geometric cooling schedule, and the SA analogy to physical annealing
- **Cite when:** Explaining the SA algorithm design choices — why we accept worse solutions with probability `exp(-delta/T)`, and why geometric cooling works

### Lin & Kernighan (1973) — 2-opt
- **File:** `literature papers/Lin_Kernighan_1973_TSP_Heuristic.pdf` *(direct PDF in REFERENCES.md)*
- **Used for:** The 2-opt neighborhood operator inside SA — reversing a segment of the tour to find a better solution. This is the move structure in `simulated_annealing.py`
- **Cite when:** Explaining why 2-opt is used as the SA neighborhood operator

### Larranaga et al. (1999) — GA for TSP
- **File:** `literature papers/Larranaga_1999_GA_TSP_Review.pdf` *(free on ResearchGate — see REFERENCES.md)*
- **Used for:** The GA implementation in `src/algorithms/genetic_algorithm.py` — specifically Order Crossover (OX), tournament selection, and swap mutation. Follow Section 3 (OX crossover) and Section 4 (mutation) when implementing
- **Cite when:** Justifying the choice of OX crossover over other crossover operators for TSP

---

> **For Claude Code:** When implementing any algorithm, open the corresponding paper alongside the code. The papers define the exact algorithm behavior — don't deviate without good reason. All implementation choices should be traceable back to one of these references.

---

## 11. Quick Reference — Known Optimal Tours

| Instance | Cities | Optimal Length |
|----------|--------|----------------|
| eil51    | 51     | 426            |
| berlin52 | 52     | 7542           |
| ch130    | 130    | 6110           |
| d198     | 198    | 15780          |

Gap formula: `gap = (found - optimal) / optimal * 100`  
Target for SA/GA reference implementations: **< 5% gap** on eil51/berlin52, **< 10%** on larger instances.

---

*Last updated: May 2026 — Denis Hoti, KU Ingolstadt*
