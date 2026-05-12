"""
GHZ-based protocol: hardware execution (ibm_fez)

This script:
- builds GHZ circuits (B=2)
- transpiles them for ibm_fez
- executes them using SamplerV2
- extracts mean_P values
- saves per-label CSV pools for analysis
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


def build_Bx6_circuit(label: str, B: int = 2) -> QuantumCircuit:
    if B != 2:
        raise ValueError("This minimal hardware script is fixed to B=2.")

    n_qubits = 6 * B
    qreg = QuantumRegister(n_qubits, "q")
    creg = ClassicalRegister(n_qubits, "meas")
    qc = QuantumCircuit(qreg, creg)

    prepare_unknown(qc, label)

    # Fanout q0 to the parent of block 2
    qc.cx(0, 6)

    # Expand each parent into its 6-qubit block
    for b in range(B):
        parent = 6 * b
        for anc in range(parent + 1, parent + 6):
            qc.cx(parent, anc)

    # Measure all qubits in the X basis
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


def parse_global_parity(bitstring: str, B: int = 2) -> int:
    bits = bitstring[::-1]
    P_tot = 1

    for b in range(B):
        local_bits = bits[6 * b : 6 * b + 6]
        sig = signature_from_local_bits(local_bits)
        p = signature_parity(sig)
        P_tot *= p

    return P_tot


def counts_to_mean_P(counts: dict[str, int], B: int = 2) -> float:
    total = 0
    nshots = 0
    for bitstring, count in counts.items():
        total += parse_global_parity(bitstring, B=B) * count
        nshots += count
    return total / nshots


def compile_circuits_for_backend(backend):
    labels = ["0", "1", "+", "-"]
    circuits = {label: build_Bx6_circuit(label, B=2) for label in labels}

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1,
    )

    return {label: pm.run(circuits[label]) for label in labels}


def label_to_filename(label: str) -> str:
    return {
        "0": "ghz_B2_pool_0.csv",
        "1": "ghz_B2_pool_1.csv",
        "+": "ghz_B2_pool_plus.csv",
        "-": "ghz_B2_pool_minus.csv",
    }[label]


def get_counts_from_pub_result(pub_result, circuit: QuantumCircuit) -> dict[str, int]:
    """
    Runtime SamplerV2 stores data under classical register names.
    Prefer 'meas', but fall back safely if the register name differs.
    """
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


def main():
    # -----------------------------
    # Pilot-run settings
    # -----------------------------
    backend_name = "ibm_fez"

    # Small pilot only
    n_samples_per_label = 10
    n_shots_per_sample = 128

    outdir = Path("ibm_fez_minimal_B2_output")
    outdir.mkdir(parents=True, exist_ok=True)

    # Read credentials from environment
    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")

    if not token:
        raise RuntimeError("Set QISKIT_IBM_TOKEN in your environment before running.")

    if not instance:
        raise RuntimeError("Set QISKIT_IBM_INSTANCE in your environment before running.")

    # Connect to paid instance
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )

    backend = service.backend(backend_name)

    print(f"Using backend: {backend.name}")
    print(f"Instance: {instance}")
    print(
        f"Status: operational={backend.status().operational}, "
        f"pending_jobs={backend.status().pending_jobs}"
    )

    # -----------------------------
    # Compile once
    # -----------------------------
    isa_circuits = compile_circuits_for_backend(backend)

    for label, circ in isa_circuits.items():
        qasm_path = outdir / f"isa_B2_{label_to_filename(label).replace('.csv', '.qasm.txt')}"
        qasm_path.write_text(str(circ))

    # -----------------------------
    # Run one label at a time
    # -----------------------------
    sampler = Sampler(mode=backend)
    summary = {}

    for label in ["0", "1", "+", "-"]:
        print(f"\nRunning label {label} ...")

        circ = isa_circuits[label]
        mean_p_pool = []

        for sample_idx in range(n_samples_per_label):
            job = sampler.run([circ], shots=n_shots_per_sample)
            result = job.result()
            pub_result = result[0]

            counts = get_counts_from_pub_result(pub_result, circ)
            mean_p = counts_to_mean_P(counts, B=2)
            mean_p_pool.append(mean_p)

            print(f"  sample {sample_idx + 1}/{n_samples_per_label}: mean_P = {mean_p:.4f}")

        df = pd.DataFrame({"mean_P": mean_p_pool})
        csv_path = outdir / label_to_filename(label)
        df.to_csv(csv_path, index=False)

        summary[label] = {
            "n_samples": n_samples_per_label,
            "shots_per_sample": n_shots_per_sample,
            "mean_of_mean_P": float(df["mean_P"].mean()),
            "std_of_mean_P": float(df["mean_P"].std(ddof=1)) if len(df) > 1 else 0.0,
            "csv_file": csv_path.name,
        }

    summary_path = outdir / "ghz_B2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nDone.")
    print(f"Saved outputs in: {outdir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
