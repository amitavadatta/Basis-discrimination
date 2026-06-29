# Ideal simulation

This directory contains the noiseless statevector simulations of the graph-state basis-discrimination protocol.

## Files

### Script

`ideal_simulation.py`

Generates all ideal simulation data and figures.

### Data

| File                               | Description                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| `ideal_pair_forbidden_rates.csv`   | Pair-level forbidden-signature probabilities for all input pairs.                |
| `ideal_batch_variance_samples.csv` | Batch statistics generated from repeated random sampling.                        |
| `ideal_threshold_scan.csv`         | Classification accuracy as a function of variance threshold.                     |
| `ideal_best_thresholds.csv`        | Optimal threshold and corresponding classification accuracy for each batch size. |

### Figures

| File                                 | Description                                                               |
| ------------------------------------ | ------------------------------------------------------------------------- |
| `ideal_pair_forbidden_rates.png`     | Pair-level forbidden-signature rates (paper figure).                      |
| `ideal_variance_histograms.png`      | Distribution of empirical batch variances (paper figure).                 |
| `ideal_variance_separation_vs_k.png` | Mean batch variance versus batch size (additional validation figure).     |
| `ideal_accuracy_vs_k.png`            | Classification accuracy versus batch size (additional validation figure). |

The first two figures are reproduced in the manuscript. The remaining two figures are retained for completeness and reproducibility.

