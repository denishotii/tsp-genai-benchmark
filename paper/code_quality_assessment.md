# Qualitative Code-Quality Assessment

Companion to the quantitative results (`results/comparison_table.csv`). Covers
the CLAUDE.md §5.4 *code-quality* dimension — spec adherence, readability,
robustness, reproducibility — that the numbers alone don't capture.

## Method

A systematic `grep` over all 72 run-directory solvers
(`genai_experiments/generated_code/{model}/{strategy}/run{1,2,3}/`) for
spec-adherence signals (neighborhood operators, local search, seeding,
multi-start), followed by a close read of representative full implementations
of each algorithm from each model (zero-shot, run 1).

## 1. Spec adherence

| Algorithm | Spec asked for | What the models did |
|---|---|---|
| **Greedy** | single-start nearest-neighbor | **Faithful** — both models, 0/12 runs added multi-start. A simple, unambiguous spec leaves no room to embellish. |
| **SA** | 2-opt neighborhood + geometric cooling | **Faithful core** — both used 2-opt/segment-reversal in 12/12 runs, geometric cooling, no off-spec neighborhoods. GPT adds extras *around* the core (below). |
| **GA** | *pure* GA: OX + swap mutation + tournament + elitism | **Not faithful — but asymmetrically.** 11/12 runs per model went beyond "pure." The *type* of enhancement differs by model and is the key finding (below). |

**The GA finding is the headline.** When asked for a pure GA, neither model
delivers one — they reflexively improve it. But *how* they improve diverges:

- **Claude** typically adds **nearest-neighbor population seeding** while
  keeping the GA core pure (no local search). Example:
  `claude/zero_shot/run1/genetic_algorithm.py` seeds ~10% of the population
  with NN tours, then runs textbook OX + swap + tournament + elitism. This is
  a mild enhancement; the GA still suffers the pure-GA scaling weakness.
- **GPT** typically builds a **true memetic GA** — a real `_two_opt()` local
  search applied to individuals. Example:
  `gpt/zero_shot/run1/genetic_algorithm.py` defines a multi-pass 2-opt descent
  and applies it within the evolutionary loop. This is a strong enhancement
  and is the mechanistic reason GPT's GA outperforms Claude's.

## 2. Readability & structure

- **Claude — clean and textbook.** Smaller, well-decomposed helper functions
  that map directly onto the named algorithm; easy to read against the spec.
  Uses the standard-library `random`. Prioritizes clarity over raw performance.
- **GPT — maximalist and defensive.** Longer, more helper functions, hybrid
  techniques, and adaptive parameter scaling by instance size. Higher quality
  but more complex and harder to map back to a single textbook algorithm.
  Uses NumPy's `default_rng`. Example: `gpt/zero_shot/run1/simulated_annealing.py`
  is ~196 lines combining exact brute-force (n ≤ 9), multi-start NN init,
  2-opt descent, SA, and a final 2-opt polish.

## 3. Robustness

- Both handle small-instance edge cases (`n = 0, 1, 2`); GPT goes further with
  an **exact brute-force branch for n ≤ 9**.
- Both recompute / verify the final tour length before returning (good
  practice — and the reason none of them could "lie" about length).
- GPT's adaptive iteration budgets scale with `n`, which helps quality but is
  double-edged: an over-budgeted GPT chain-of-thought GA was the **only**
  reliability failure in the whole matrix — it timed out (>60 s) on the two
  largest instances (ch130, d198) in one run while running fine elsewhere.
  *Correct but non-scaling in runtime.*

## 4. Reproducibility

- **GPT seeds its RNG** (e.g. `np.random.default_rng(123456789)`), so its
  generated solvers are deterministic across executions.
- **Claude often does not seed** (e.g. SA uses `random.randint` with no seed),
  so its generated solvers vary run-to-run. Per CLAUDE.md §9 (reproducibility),
  GPT's output is the better-behaved here.

## 5. GPT-5.5 vs Claude Opus 4.7 — distilled

| Dimension | GPT-5.5 | Claude Opus 4.7 |
|---|---|---|
| Default posture | Aggressive optimizer-engineer | Clean textbook implementer |
| Enhancements | Hybridizes freely (memetic GA, SA + 2-opt + exact-small) | Mild (NN seeding); keeps cores pure |
| Readability | More complex, longer | Clearer, shorter, spec-mappable |
| Robustness | More defensive; occasionally over-engineers (timeout) | Adequate; simpler |
| Reproducibility | Seeds RNG | Often unseeded |
| Net quality | Higher and more consistent | Decent but more variable |

## Link to the quantitative results

The code reading *explains* the numbers rather than restating them:

- GPT's lower, tighter SA/GA gaps ⇐ it adds local search (memetic GA) and
  polishing that the spec didn't ask for.
- Claude's higher-variance GA ⇐ it keeps the GA pure (only seeded), inheriting
  the pure-GA scaling weakness our own reference implementation showed.
- The lone reliability failure ⇐ GPT's adaptive budgeting over-scaled on large
  instances in one generation.
- Identical greedy results across all cells ⇐ a deterministic algorithm with a
  faithful, single-start implementation from both models.

**Overarching observation for the paper:** the more under-specified the
algorithm (GA: "use a genetic algorithm"), the more the models substitute their
own engineering judgment — often producing *better-than-asked* solvers. The
more precisely specified the task (greedy, SA neighborhood), the more faithfully
they follow it. "Can LLMs implement these algorithms?" is therefore entangled
with "will they implement *exactly* the algorithm asked, or a better one?" —
and here, both models lean toward the latter whenever the spec leaves room.
