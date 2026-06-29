# IBM Fez experimental results

This directory contains the experimental implementation of the graph-state basis-discrimination protocol on the IBM Quantum processor **ibm_fez**.

## Hardware script

`run_graph_state_ibm_fez.py`

Constructs the graph-state circuits, submits the jobs to IBM Quantum, retrieves the results, and performs the statistical analysis.

## Experimental data

| File                             | Description                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| `graph_random_pair_pools.csv`    | Raw pair-level experimental data collected from IBM Fez.            |
| `graph_random_pair_summary.json` | Summary statistics for all input-pair labels.                       |
| `batch_statistics.csv`           | Batch-level forbidden-signature statistics used for classification. |

## Variance-based analysis

| File                           | Description                                          |
| ------------------------------ | ---------------------------------------------------- |
| `variance_threshold_scan.csv`  | Classification accuracy for all variance thresholds. |
| `variance_best_thresholds.csv` | Optimal variance threshold for each batch size.      |

## Additional MAD analysis

The repository also contains an alternative analysis based on the mean absolute deviation (MAD):

* `mad05_threshold_scan.csv`
* `mad05_best_thresholds.csv`
* `mad05_separation.png`
* `mad05_threshold_scan.png`

These files are retained for completeness but are not used in the final manuscript.

## Figures

| File                          | Description                                                                         |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| `pair_label_scores.png`       | Pair-level forbidden-signature rates (paper figure).                                |
| `variance_separation.png`     | Batch variance separation between Z- and X-basis ensembles (paper figure).          |
| `variance_best_accuracy.png`  | Classification accuracy obtained using the variance-based estimator (paper figure). |
| `variance_threshold_scan.png` | Threshold optimisation (additional validation figure).                              |

## ISA circuits

The directory

```
isa_circuits/
```

contains the transpiled ISA circuits executed on **ibm_fez** for all eight input-pair labels. These circuits illustrate the shallow-depth implementation used in the hardware experiments.

