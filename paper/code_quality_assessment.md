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
  double-edged: an over-budgeted GPT chain-of-thought GA accounts for **all 3
  reliability failures in the 432-row matrix** — the *same generated solver*
  (run 2) timed out on `ch130` (60 s), `d198` (60 s), *and* `pr1002` (300 s),
  while runs 1 and 3 of the same cell completed every instance under 66 s.
  *Correct algorithmically, but non-scaling in runtime — and the failure mode
  is generation-specific rather than systematic.*

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

## 6. What the larger instances (pr439, pr1002) revealed about scaling

The original ladder (51 → 198 cities) confirmed that LLM-generated code *runs*.
The extended ladder (439 → 1002 cities) — added 2026-06-24 on supervisor
feedback — confirmed something different: *that it scales unevenly*. The same
artifacts were re-tested (no new generations); the only changes were the
problem size and a per-instance wall-clock budget (300 s on the 439+ city
instances vs. 60 s below).

### Code-scaling differentiation (new at scale)

- **Greedy stays trivial.** Same code, same gap, every cell — `pr439: 22.44 %`,
  `pr1002: 27.82 %`, identical across all 12 LLM cells per instance. Confirms
  the spec-prescribed deterministic nature of nearest-neighbor with single start.
- **SA splits.** At ≤ 198 cities every LLM-generated SA is within a few percent
  of the reference. At `pr1002` the *best* LLM SA (Claude CoT, **6.28 %**) beats
  the reference SA (9.97 %), but the *mean* LLM SA at `pr1002` is 17–23 % —
  worse than the reference. Scale exposes both directions of variance that
  smaller instances hid.
- **GA is the dramatic one.** Reference pure GA collapses to **915 % gap** at
  `pr1002` (essentially a random tour). The best LLM GA at `pr1002` is **9.80 %**
  (GPT iterative refinement). The mechanism is exactly the §1 finding —
  memetic-style enhancements — but at 1000 cities the upgrade is the difference
  between unusable and competitive. The "*useful but not faithful*" tension is
  no longer subtle.

### Runtime-scaling failure mode (new at scale)

The lone failing solver from §3 (GPT CoT GA, run 2) has a runtime profile that
documents exactly how generated code degrades:

| Instance | Cities | Runtime | Status |
|---|---|---|---|
| eil51 | 51 | 28.4 s | ok |
| berlin52 | 52 | 29.1 s | ok |
| ch130 | 130 | 60.0 s | **timeout** |
| d198 | 198 | 60.0 s | **timeout** |
| pr439 | 439 | 262.1 s | ok *(just under the 300 s wall)* |
| pr1002 | 1002 | 300.0 s | **timeout** |

Same algorithm specification, same model, same prompt — but this *particular*
generation produces code whose work grows non-linearly with `n`. Other runs of
the *exact same cell* completed every instance under 66 s. **The failure is in
the generation, not in the algorithm.** This is the single most concrete piece
of evidence in the study that LLM code-quality variance is the load-bearing
risk, not LLM code-correctness.

### Variance quantified by scale

Variance was hard to read at small scale: most cells hit single-digit gaps and
small spreads. The extended ladder makes the variance visible. Claude few-shot
GA, for example, has a standard deviation of **152 %** in gap across its 18
instance-runs — best 0.72 % on `berlin52`, worst 600 %+ on `pr1002` — same
prompt, same model, same strategy. The reproducibility-risk story moves from
*plausible argument* (§1's "different LLMs add different enhancements") to
*hard number* once the instance is large enough to expose the difference
between adding 2-opt and not.

---

## Link to the quantitative results

The code reading *explains* the numbers rather than restating them:

- GPT's lower, tighter SA/GA gaps ⇐ it adds local search (memetic GA) and
  polishing that the spec didn't ask for.
- Claude's higher-variance GA ⇐ it keeps the GA pure (only seeded), inheriting
  the pure-GA scaling weakness our own reference implementation showed.
- All 3 reliability failures ⇐ a single GPT CoT GA generation whose runtime
  grows non-linearly with `n` — generation-specific, not systematic.
- Identical greedy results across all cells ⇐ a deterministic algorithm with a
  faithful, single-start implementation from both models.
- The dramatic 915 % → 9.8 % gap closure on `pr1002` ⇐ the LLM-added 2-opt
  local search inside a memetic GA, which the spec did not request but which
  is the only thing keeping the population-based metaheuristic competitive at
  scale.

**Overarching observation for the paper:** the more under-specified the
algorithm (GA: "use a genetic algorithm"), the more the models substitute their
own engineering judgment — often producing *better-than-asked* solvers. The
more precisely specified the task (greedy, SA neighborhood), the more faithfully
they follow it. "Can LLMs implement these algorithms?" is therefore entangled
with "will they implement *exactly* the algorithm asked, or a better one?" —
and here, both models lean toward the latter whenever the spec leaves room.

Scale doesn't change this conclusion; it sharpens it.
