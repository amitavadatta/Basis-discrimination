# Basis-discrimination
This is code and data repository for the paper "Beyond Single-Copy Quantum Measurements: Ensemble Structure Enables Basis Discrimination"


# Beyond Single-Copy Quantum Measurements: Ensemble Structure Enables Basis Discrimination

This repository contains the code and representative datasets used in the paper:

> *Beyond Single-Copy Quantum Measurements: Ensemble Structure Enables Basis Discrimination*

The work demonstrates that quantum ensembles that are indistinguishable at the level of their single-copy density matrix can be distinguished using multi-copy structure. Two experimentally realizable protocols are implemented:

- **GHZ-based protocol** (global parity measurement)
- **Graph-state protocol** (higher-order statistical signatures)

All experiments were performed on IBM Quantum hardware (`ibm_fez`) using Qiskit.

---

# 📁 Repository structure

```text
.
├── README.md
├── GHZ/
│   ├── run_ghz_b2_ibm_fez.py
│   ├── analyze_ghz_b2_results.py
│   └── data/
│       ├── ghz_B2_pool_0.csv
│       ├── ghz_B2_pool_1.csv
│       ├── ghz_B2_pool_plus.csv
│       └── ghz_B2_pool_minus.csv
│
├── Graph_State/
│   ├── run_graph_ibm_fez.py
│   └── data/
│       ├── graph_random_pair_pools.csv
│       ├── variance_threshold_scan.csv
│       └── variance_best_thresholds.csv

#Requrements

```bash
pip install qiskit qiskit-ibm-runtime numpy pandas matplotlib

#GHZ protocol - Hardware Execution
python GHZ/run_ghz_b2_ibm_fez.py

This:

- builds GHZ circuits (12 qubits, B=2)
- executes on ibm_fez
- generates per-label datasets (mean_P)

#Analysis and plots for the GHZ protocol

python GHZ/analyze_ghz_b2_results.py

Generates:

single-instance histograms
threshold scans
batch classification accuracy plots

#Gaph-state protocol 
python Graph_State/run_graph_ibm_fez.py

This:

- constructs graph-state circuits for all input pairs
- executes them on ibm_fez
- extracts forbidden-signature statistics
- performs batch resampling under the same-basis constraint
- generates:
    - separation plots
    - threshold scans
    - classification accuracy


