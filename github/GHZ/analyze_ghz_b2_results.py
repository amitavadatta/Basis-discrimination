"""
GHZ-based protocol: hardware data analysis

This script processes experimental data obtained from the GHZ-based protocol
executed on the ibm_fez quantum processor.

Functionality:
- Loads per-label hardware datasets (mean_P values) for input states {0, 1, +, −}
- Performs single-instance classification using thresholding on mean_P
- Implements batch X-vs-Z discrimination using the second-moment statistic S^2
- Computes classification accuracy and threshold scans
- Generates all figures reported in the Results section

Input:
- CSV files containing mean_P values for each input label

Output:
- Processed CSV files with accuracy metrics
- Plots: histograms, threshold scans, batch accuracy, and S^2 statistics

Notes:
- The analysis relies on repeated sampling from hardware-generated datasets
- Results are robust to moderate statistical fluctuations due to finite sampling
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


PAIR_INDEX = [(0, 1), (2, 3), (4, 5)]


def prepare_unknown(qc: QuantumCircuit, label: str) -> None:
    if label == "0":
        return
    if label == "1":
        qc.x(0)
    elif label == "+":
        qc.h(0)
    elif label == "-":
        qc.x(0)
        qc.h(0)
    else:
        raise ValueError("label must be one of '0', '1', '+', '-'.")


def build_Bx6_circuit(label: str, B: int) -> QuantumCircuit:
    if B < 2:
        raise ValueError("B must be at least 2.")

    n_qubits = 6 * B
    qreg = QuantumRegister(n_qubits, "q")
    creg = ClassicalRegister(n_qubits, "meas")
    qc = QuantumCircuit(qreg, creg)

    # Prepare unknown on parent of block 0
    prepare_unknown(qc, label)

    # Fanout from parent of block 0 to parents of remaining blocks
    for b in range(1, B):
        qc.cx(0, 6 * b)

    # Expand each parent into its local 6-qubit block
    for b in range(B):
        parent = 6 * b
        for anc in range(parent + 1, parent + 6):
            qc.cx(parent, anc)

    # Measure all qubits in X basis
    for q in range(n_qubits):
        qc.h(q)
        qc.measure(q, q)

    return qc


def bit_to_xeig(bit: str) -> int:
    return +1 if bit == "0" else -1


def signature_from_local_bits(bitstring: str):
    xeigs = [bit_to_xeig(b) for b in bitstring]
    return tuple(xeigs[i] * xeigs[j] for i, j in PAIR_INDEX)


def signature_parity(sig) -> int:
    p = 1
    for s in sig:
        p *= s
    return p


def parse_global_parity(bitstring: str, B: int) -> int:
    bits = bitstring[::-1]
    P_tot = 1

    for b in range(B):
        local_bits = bits[6 * b: 6 * b + 6]
        sig = signature_from_local_bits(local_bits)
        p = signature_parity(sig)
        P_tot *= p

    return P_tot


def counts_to_mean_P(counts: dict[str, int], B: int) -> float:
    total = 0
    nshots = 0
    for bitstring, count in counts.items():
        total += parse_global_parity(bitstring, B=B) * count
        nshots += count
    return total / nshots


def compile_circuits_for_backend(backend, B: int):
    labels = ["0", "1", "+", "-"]
    circuits = {label: build_Bx6_circuit(label, B=B) for label in labels}

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1,
    )

    return {label: pm.run(circuits[label]) for label in labels}


def label_to_filename(label: str, B: int) -> str:
    return {
        "0": f"ghz_B{B}_pool_0.csv",
        "1": f"ghz_B{B}_pool_1.csv",
        "+": f"ghz_B{B}_pool_plus.csv",
        "-": f"ghz_B{B}_pool_minus.csv",
    }[label]


def get_counts_from_pub_result(pub_result, circuit: QuantumCircuit) -> dict[str, int]:
    if hasattr(pub_result.data, "meas"):
        return pub_result.data.meas.get_counts()

    creg_names = [creg.name for creg in circuit.cregs]
    if len(creg_names) == 1:
        reg_name = creg_names[0]
        data_obj = getattr(pub_result.data, reg_name, None)
        if data_obj is None:
            raise RuntimeError(
                f"Result data has no attribute '{reg_name}'. "
                f"Available attributes: {dir(pub_result.data)}"
            )
        return data_obj.get_counts()

    raise RuntimeError(
        f"Could not determine classical register. "
        f"Circuit cregs={creg_names}, result attrs={dir(pub_result.data)}"
    )


def run_for_B(
    B: int,
    backend_name: str,
    n_samples_per_label: int,
    n_shots_per_sample: int,
):
    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")

    if not token:
        raise RuntimeError("Set QISKIT_IBM_TOKEN in your environment before running.")
    if not instance:
        raise RuntimeError("Set QISKIT_IBM_INSTANCE in your environment before running.")

    outdir = Path(f"ibm_fez_B{B}_pilot_output")
    outdir.mkdir(parents=True, exist_ok=True)

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )

    backend = service.backend(backend_name)

    print(f"\n=== Running B={B} on {backend.name} ===")
    print(f"Instance: {instance}")
    print(
        f"Status: operational={backend.status().operational}, "
        f"pending_jobs={backend.status().pending_jobs}"
    )

    isa_circuits = compile_circuits_for_backend(backend, B=B)

    for label, circ in isa_circuits.items():
        qasm_path = outdir / f"isa_B{B}_{label_to_filename(label, B).replace('.csv', '.qasm.txt')}"
        qasm_path.write_text(str(circ))

    sampler = Sampler(mode=backend)
    summary = {}

    for label in ["0", "1", "+", "-"]:
        print(f"\nRunning label {label} for B={B} ...")

        circ = isa_circuits[label]
        mean_p_pool = []

        for sample_idx in range(n_samples_per_label):
            job = sampler.run([circ], shots=n_shots_per_sample)
            result = job.result()
            pub_result = result[0]

            counts = get_counts_from_pub_result(pub_result, circ)
            mean_p = counts_to_mean_P(counts, B=B)
            mean_p_pool.append(mean_p)

            print(f"  sample {sample_idx + 1}/{n_samples_per_label}: mean_P = {mean_p:.6f}")

        df = pd.DataFrame({"mean_P": mean_p_pool})
        csv_path = outdir / label_to_filename(label, B)
        df.to_csv(csv_path, index=False)

        summary[label] = {
            "B": B,
            "n_samples": n_samples_per_label,
            "shots_per_sample": n_shots_per_sample,
            "mean_of_mean_P": float(df["mean_P"].mean()),
            "std_of_mean_P": float(df["mean_P"].std(ddof=1)) if len(df) > 1 else 0.0,
            "csv_file": csv_path.name,
        }

    summary_path = outdir / f"ghz_B{B}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nDone for B={B}.")
    print(f"Saved outputs in: {outdir.resolve()}")
    print(json.dumps(summary, indent=2))


def main():
    backend_name = "ibm_fez"

    # Choose one:
    #B = 2
     B = 3

    # Suggested pilots:
    # For B=2:  n_samples_per_label = 30, n_shots_per_sample = 128
    # For B=3:  n_samples_per_label = 10, n_shots_per_sample = 128
    n_samples_per_label = 10
    n_shots_per_sample = 128

    run_for_B(
        B=B,
        backend_name=backend_name,
        n_samples_per_label=n_samples_per_label,
        n_shots_per_sample=n_shots_per_sample,
    )


if __name__ == "__main__":
    main()
