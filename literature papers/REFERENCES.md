# Literature References — TSP + GenAI Benchmark

> Download each paper as a PDF and save it in this folder alongside this file.
> Papers marked 🔓 are freely available via direct link. Papers marked 🔒 require library access.

---

## 1. TSPLIB — Benchmark Instances

**Reinelt, G. (1991)**
*TSPLIB — A Traveling Salesman Problem Library*
ORSA Journal on Computing, Vol. 3, No. 4, pp. 376–384.
DOI: [10.1287/ijoc.3.4.376](https://doi.org/10.1287/ijoc.3.4.376)

🔒 **Access:** Behind INFORMS paywall — use your KU Ingolstadt library portal.
Search for: `TSPLIB A Traveling Salesman Problem Library Reinelt 1991`

**Why read it:** Describes the four benchmark instances used in this project (eil51, berlin52, ch130, d198), the `.tsp` file format, and how optimal tour lengths were established.

**Key sections:** Instance description format, EUC_2D distance specification, known optimal values.

---

## 2. Simulated Annealing

**Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983)**
*Optimization by Simulated Annealing*
Science, Vol. 220, No. 4598, pp. 671–680.
DOI: [10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671)

🔓 **Direct PDF:** https://mk.bcgsc.ca/papers/kirkpatrick-simulatedannealing.pdf

**Why read it:** The original SA paper. Introduces the Metropolis criterion, the cooling schedule analogy, and applies SA directly to TSP. This is what you cite when you explain why SA can escape local optima.

**Key sections:** Sections 1–3 (combinatorial optimization, the annealing analogy, Metropolis algorithm). The TSP application section at the end is directly relevant.

---

## 3. 2-opt Neighborhood / TSP Heuristics

**Lin, S., & Kernighan, B. W. (1973)**
*An Effective Heuristic Algorithm for the Traveling-Salesman Problem*
Operations Research, Vol. 21, No. 2, pp. 498–516.
DOI: [10.1287/opre.21.2.498](https://doi.org/10.1287/opre.21.2.498)

🔓 **Direct PDF:** https://www.cs.princeton.edu/~bwk/btl.mirror/tsp.pdf

**Why read it:** Defines the 2-opt move (reversing a segment of the tour) which is the neighborhood operator used inside the SA implementation. Also gives intuition for why local search on TSP works.

**Key sections:** Section 2 (the basic move structure), Section 3 (2-opt specifically).

---

## 4. Genetic Algorithm for TSP (Order Crossover)

**Larranaga, P., Kuijpers, C. M. H., Murga, R. H., Inza, I., & Dizdarevic, S. (1999)**
*Genetic Algorithms for the Travelling Salesman Problem: A Review of Representations and Operators*
Artificial Intelligence Review, Vol. 13, pp. 129–170.

🔓 **Search:** Google Scholar → `"Larranaga 1999 genetic algorithms travelling salesman review"`
Free PDF widely available on ResearchGate and university repositories.

**Why read it:** Comprehensive review of GA representations for TSP, including a clear description of Order Crossover (OX), which is the crossover operator used in this project. Also covers tournament selection and mutation operators.

**Key sections:** Section 3 (crossover operators — focus on OX), Section 4 (mutation operators — focus on swap mutation).

---

## Summary Table

| # | Paper | Year | Topic | Access |
|---|-------|------|-------|--------|
| 1 | Reinelt | 1991 | TSPLIB benchmark instances | 🔒 KU Library |
| 2 | Kirkpatrick et al. | 1983 | Simulated Annealing | 🔓 Direct PDF |
| 3 | Lin & Kernighan | 1973 | 2-opt moves / TSP heuristics | 🔓 Direct PDF |
| 4 | Larranaga et al. | 1999 | GA for TSP, OX crossover | 🔓 ResearchGate |

---

## How to Cite (APA 7th)

```
Reinelt, G. (1991). TSPLIB—A traveling salesman problem library.
  ORSA Journal on Computing, 3(4), 376–384.

Kirkpatrick, S., Gelatt, C. D., Jr., & Vecchi, M. P. (1983).
  Optimization by simulated annealing. Science, 220(4598), 671–680.

Lin, S., & Kernighan, B. W. (1973). An effective heuristic algorithm
  for the traveling-salesman problem. Operations Research, 21(2), 498–516.

Larranaga, P., Kuijpers, C. M. H., Murga, R. H., Inza, I., &
  Dizdarevic, S. (1999). Genetic algorithms for the travelling salesman
  problem: A review of representations and operators.
  Artificial Intelligence Review, 13, 129–170.
```

---

*Project: TSP + GenAI Benchmark · Denis Hoti · KU Ingolstadt · SS 2026*
