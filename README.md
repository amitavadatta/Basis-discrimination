# Basis Discrimination via Graph-State Stabilizer Measurements

This repository contains the reference implementation accompanying the manuscript

> **Beyond Single-Copy Quantum Measurements: Ensemble Structure Enables Basis Discrimination**

The repository provides both ideal (noise-free) simulations and experimental results obtained on the IBM Quantum superconducting processor **ibm_fez**.

## Repository structure

```
Basis-discrimination/
│
├── README.md
├── ideal_simulation/
└── graph_random_same_basis_fez_results/
```

### `ideal_simulation/`

Ideal statevector simulations of the graph-state protocol.

Contents include

* simulation scripts;
* generated CSV data;
* figures reproduced in the manuscript;
* additional validation figures retained for reproducibility.

### `graph_random_same_basis_fez_results/`

Experimental implementation on the IBM Quantum **ibm_fez** processor.

Contents include

* hardware execution script;
* raw pair-level data;
* processed batch statistics;
* threshold scans;
* ISA-transpiled circuits;
* figures reproduced in the manuscript.

---

## Figures

### Figures appearing in the manuscript

### Ideal simulations

| Figure                                  | File                                 |
| --------------------------------------- | ------------------------------------ |
| Pair-level forbidden-signature response | `ideal_pair_forbidden_rates.png`     |
| Batch variance histograms               | `ideal_variance_histograms.png`      |

### IBM Fez experiments

| Figure                                  | File                         |
| --------------------------------------- | ---------------------------- |
| Pair-level forbidden-signature response | `pair_label_scores.png`      |
| Batch variance separation               | `variance_separation.png`    |
| Classification accuracy                 | `variance_best_accuracy.png` |

---

## Additional validation figures

The repository also contains several figures that are not reproduced in the manuscript but were generated during development and validation of the protocol.

These include

* threshold scans;
* accuracy optimisation;
* alternative batch statistics.

They are retained to facilitate independent verification and further analysis.

---

## Software

The code is written in Python using

* Qiskit
* Qiskit Runtime
* NumPy
* SciPy
* Pandas
* Matplotlib

---

## Citation

If you use this software or build upon this work, please cite the accompanying manuscript.

