"""
Graph-state protocol: hardware execution and analysis (ibm_fez)

This script:
- builds graph-state circuits for all input pairs
- executes them on ibm_fez using SamplerV2
- extracts forbidden-signature and parity statistics
- performs batch resampling under the same-basis promise
- evaluates variance- and MAD-based classifiers
- generates all figures reported in the Results section

Inputs:
- pair labels: {00, 01, 10, 11, ++, +-, -+, --}

Outputs:
- CSV: per-pair forbidden rates
- CSV: batch statistics
- CSV: threshold scans and best thresholds
- plots: separation, threshold scan, classification accuracy
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


# ============================================================
# Pair labels
# ============================================================

Z_PAIR_LABELS = ["00", "01", "10", "11"]
X_PAIR_LABELS = ["++", "+-", "-+", "--"]
ALL_PAIR_LABELS = Z_PAIR_LABELS + X_PAIR_LABELS


# Same-state Z reference signatures from earlier analysis
Z_SIGNATURES_SAME_STATE = {
    (+1, +1, +1),
    (-1, -1, +1),
}

EVEN_SIGNATURES = {
    (+1, +1, +1),
    (+1, -1, -1),
    (-1, +1, -1),
    (-1, -1, +1),
}

FORBIDDEN_FOR_SAME_STATE_Z = EVEN_SIGNATURES - Z_SIGNATURES_SAME_STATE


# ============================================================
# Circuit construction
# q0 = q1 unknown
# q1 = q2 unknown
# q2 = a1
# q3 = a2
# q4 = a3
# ============================================================

def prepare_single(qc: QuantumCircuit, qubit: int, label: str) -> None:
    if label == "0":
        return
    elif label == "1":
        qc.x(qubit)
    elif label == "+":
        qc.h(qubit)
    elif label == "-":
        qc.x(qubit)
        qc.h(qubit)
    else:
        raise ValueError(f"Bad label {label}")


def prepare_pair(qc: QuantumCircuit, pair_label: str) -> None:
    prepare_single(qc, 0, pair_label[0])
    prepare_single(qc, 1, pair_label[1])


def build_graph_circuit(pair_label: str) -> QuantumCircuit:
    q = QuantumRegister(5, "q")
    c = ClassicalRegister(3, "meas")
    qc = QuantumCircuit(q, c)

    prepare_pair(qc, pair_label)

    # Ancillas in |+>
    qc.h(2)
    qc.h(3)
    qc.h(4)

    # Initial CNOT
    qc.cx(0, 1)

    # Graph-state CZ layer
    qc.cz(0, 2)  # q1-a1
    qc.cz(0, 3)  # q1-a2
    qc.cz(1, 3)  # q2-a2
    qc.cz(1, 4)  # q2-a3

    # X-basis measurement of ancillas
    qc.h(2)
    qc.h(3)
    qc.h(4)

    qc.measure(2, 0)
    qc.measure(3, 1)
    qc.measure(4, 2)

    return qc


# ============================================================
# Counts parsing
# ============================================================

def bit_to_eig(bit: str) -> int:
    return +1 if bit == "0" else -1


def signature_from_bitstring(bitstring: str):
    bits = bitstring[::-1]
    return (
        bit_to_eig(bits[0]),  # a1
        bit_to_eig(bits[1]),  # a2
        bit_to_eig(bits[2]),  # a3
    )


def counts_to_forbidden_rate(counts: dict[str, int]) -> float:
    total = 0
    forbidden = 0

    for bitstring, count in counts.items():
        sig = signature_from_bitstring(bitstring)
        if sig in FORBIDDEN_FOR_SAME_STATE_Z:
            forbidden += count
        total += count

    return forbidden / total


def counts_to_even_rate(counts: dict[str, int]) -> float:
    total = 0
    even = 0

    for bitstring, count in counts.items():
        sig = signature_from_bitstring(bitstring)
        if sig[0] * sig[1] * sig[2] == +1:
            even += count
        total += count

    return even / total


def get_counts_from_pub_result(pub_result, circuit):
    """
    Robust SamplerV2 counts extraction.
    For this circuit, the classical register is named 'meas'.
    """
    if hasattr(pub_result.data, "meas"):
        return pub_result.data.meas.get_counts()

    reg_name = circuit.cregs[0].name
    return getattr(pub_result.data, reg_name).get_counts()


# ============================================================
# IBM Fez hardware run
# ============================================================

def build_pools_ibm_fez(
    n_samples_per_pair_label=20,
    shots_per_sample=128,
    outdir="graph_random_same_basis_fez_results",
    optimization_level=1,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("QISKIT_IBM_TOKEN")
    instance = os.environ.get("QISKIT_IBM_INSTANCE")

    if not token:
        raise RuntimeError("Please set QISKIT_IBM_TOKEN.")
    if not instance:
        raise RuntimeError("Please set QISKIT_IBM_INSTANCE.")

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )

    backend = service.backend("ibm_fez")
    sampler = Sampler(mode=backend)

    circuits = {lab: build_graph_circuit(lab) for lab in ALL_PAIR_LABELS}
    compiled = {
        lab: transpile(circ, backend, optimization_level=optimization_level)
        for lab, circ in circuits.items()
    }

    # Save transpiled circuits for inspection
    isa_dir = outdir / "isa_circuits"
    isa_dir.mkdir(exist_ok=True)

    for lab, circ in compiled.items():
        safe = lab.replace("+", "plus").replace("-", "minus")
        (isa_dir / f"graph_isa_{safe}.txt").write_text(str(circ))

    rows = []
    summary = {}

    for lab, circ in compiled.items():
        family = "Z" if lab in Z_PAIR_LABELS else "X"
        print(f"\nRunning pair {lab}, family {family}")

        forbidden_vals = []
        even_vals = []

        for k in range(n_samples_per_pair_label):
            job = sampler.run([circ], shots=shots_per_sample)
            result = job.result()
            counts = get_counts_from_pub_result(result[0], circ)

            forbidden_rate = counts_to_forbidden_rate(counts)
            even_rate = counts_to_even_rate(counts)

            forbidden_vals.append(forbidden_rate)
            even_vals.append(even_rate)

            rows.append(
                {
                    "pair_label": lab,
                    "family": family,
                    "sample": k,
                    "shots": shots_per_sample,
                    "forbidden_rate": forbidden_rate,
                    "even_rate": even_rate,
                    "job_id": job.job_id(),
                }
            )

            print(
                f"  sample {k+1}/{n_samples_per_pair_label}: "
                f"forbidden={forbidden_rate:.4f}, even={even_rate:.4f}, job={job.job_id()}"
            )

        summary[lab] = {
            "family": family,
            "mean_forbidden_rate": float(np.mean(forbidden_vals)),
            "std_forbidden_rate": float(np.std(forbidden_vals, ddof=1)) if len(forbidden_vals) > 1 else 0.0,
            "mean_even_rate": float(np.mean(even_vals)),
            "std_even_rate": float(np.std(even_vals, ddof=1)) if len(even_vals) > 1 else 0.0,
        }

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "graph_random_pair_pools.csv", index=False)

    with open(outdir / "graph_random_pair_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    return df


# ============================================================
# Batch-statistics analysis
# ============================================================

def sample_batch_stats(df, family, n_pairs, rng):
    sub = df[df["family"] == family]
    sample = sub.sample(
        n=n_pairs,
        replace=True,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )

    vals = sample["forbidden_rate"].to_numpy(dtype=float)

    return {
        "mean": float(np.mean(vals)),
        "var": float(np.var(vals, ddof=0)),
        "mad05": float(np.mean(np.abs(vals - 0.5))),
        "range": float(np.max(vals) - np.min(vals)),
    }


def build_batch_stat_table(
    df,
    unknown_counts=(2, 4, 6, 8, 10, 12, 16, 20),
    n_batches=2000,
    seed=2027,
):
    rng = np.random.default_rng(seed)
    rows = []

    for n_unknowns in unknown_counts:
        if n_unknowns % 2 != 0:
            raise ValueError("This protocol uses two unknown qubits at a time.")

        n_pairs = n_unknowns // 2

        for family in ["Z", "X"]:
            for _ in range(n_batches):
                stats = sample_batch_stats(df, family, n_pairs, rng)
                rows.append(
                    {
                        "family": family,
                        "n_unknowns": n_unknowns,
                        "n_pairs": n_pairs,
                        **stats,
                    }
                )

    return pd.DataFrame(rows)


def threshold_scan(
    batch_df,
    stat_name="var",
    thresholds=None,
    predict_z_if_above=True,
):
    if thresholds is None:
        lo = float(batch_df[stat_name].min())
        hi = float(batch_df[stat_name].max())
        thresholds = np.linspace(lo, hi, 51)

    rows = []

    for t in thresholds:
        for n_unknowns in sorted(batch_df["n_unknowns"].unique()):
            sub = batch_df[batch_df["n_unknowns"] == n_unknowns].copy()

            if predict_z_if_above:
                sub["pred"] = np.where(sub[stat_name] > t, "Z", "X")
            else:
                sub["pred"] = np.where(sub[stat_name] < t, "Z", "X")

            acc_z = float(np.mean(sub[sub["family"] == "Z"]["pred"] == "Z"))
            acc_x = float(np.mean(sub[sub["family"] == "X"]["pred"] == "X"))
            overall = 0.5 * (acc_z + acc_x)

            rows.append(
                {
                    "threshold": float(t),
                    "n_unknowns": n_unknowns,
                    "stat": stat_name,
                    "acc_Z": acc_z,
                    "acc_X": acc_x,
                    "overall_accuracy": overall,
                }
            )

    return pd.DataFrame(rows)


def best_thresholds(scan_df):
    rows = []
    for n_unknowns in sorted(scan_df["n_unknowns"].unique()):
        sub = scan_df[scan_df["n_unknowns"] == n_unknowns]
        rows.append(sub.iloc[sub["overall_accuracy"].argmax()].to_dict())
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_pair_label_scores(df, outpath):
    order = Z_PAIR_LABELS + X_PAIR_LABELS
    means = df.groupby("pair_label")["forbidden_rate"].mean().reindex(order)
    stds = df.groupby("pair_label")["forbidden_rate"].std().reindex(order)

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(order)), means.values, yerr=stds.values, capsize=4)
    plt.xticks(range(len(order)), order)
    plt.ylabel("Mean forbidden-signature rate")
    plt.xlabel("Input pair")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.title("ibm_fez graph-state classifier: pair-label response")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_stat_separation(batch_df, stat_name, ylabel, outpath):
    plt.figure(figsize=(8, 5))

    for family in ["Z", "X"]:
        xs, means, stds = [], [], []

        for n in sorted(batch_df["n_unknowns"].unique()):
            sub = batch_df[
                (batch_df["family"] == family)
                & (batch_df["n_unknowns"] == n)
            ]
            xs.append(n)
            means.append(sub[stat_name].mean())
            stds.append(sub[stat_name].std(ddof=1))

        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=f"True {family}")

    plt.xlabel("Number of unknown qubits used")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"ibm_fez random same-basis promise: {stat_name} separation")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_threshold_scan(scan_df, outpath):
    plt.figure(figsize=(8, 5))

    for n in sorted(scan_df["n_unknowns"].unique()):
        sub = scan_df[scan_df["n_unknowns"] == n].sort_values("threshold")
        plt.plot(sub["threshold"], sub["overall_accuracy"], marker="o", label=f"N={n}")

    plt.xlabel("Threshold")
    plt.ylabel("Overall accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.title(f"ibm_fez threshold scan using {scan_df['stat'].iloc[0]}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_best_accuracy(best_df, outpath):
    plt.figure(figsize=(8, 5))

    plt.plot(best_df["n_unknowns"], best_df["overall_accuracy"], marker="o", label="overall")
    plt.plot(best_df["n_unknowns"], best_df["acc_Z"], marker="o", label="True Z")
    plt.plot(best_df["n_unknowns"], best_df["acc_X"], marker="o", label="True X")

    plt.xlabel("Number of unknown qubits used")
    plt.ylabel("Classification accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("ibm_fez best variance-classifier accuracy")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    outdir = Path("graph_random_same_basis_fez_results")
    outdir.mkdir(exist_ok=True)

    # Hardware acquisition. Start small if quota is limited.
    n_samples_per_pair_label = 20
    shots_per_sample = 128

    df = build_pools_ibm_fez(
        n_samples_per_pair_label=n_samples_per_pair_label,
        shots_per_sample=shots_per_sample,
        outdir=outdir,
        optimization_level=1,
    )

    # If you already ran hardware and only want analysis, comment above and use:
    # df = pd.read_csv(outdir / "graph_random_pair_pools.csv")

    plot_pair_label_scores(df, outdir / "pair_label_scores.png")

    unknown_counts = (2, 4, 6, 8, 10, 12, 16, 20)

    batch_df = build_batch_stat_table(
        df,
        unknown_counts=unknown_counts,
        n_batches=2000,
        seed=2027,
    )
    batch_df.to_csv(outdir / "batch_statistics.csv", index=False)

    # Variance classifier
    var_scan = threshold_scan(
        batch_df,
        stat_name="var",
        thresholds=np.linspace(0.0, 0.25, 51),
        predict_z_if_above=True,
    )
    var_best = best_thresholds(var_scan)

    var_scan.to_csv(outdir / "variance_threshold_scan.csv", index=False)
    var_best.to_csv(outdir / "variance_best_thresholds.csv", index=False)

    plot_stat_separation(
        batch_df,
        stat_name="var",
        ylabel="Variance of forbidden-signature rate",
        outpath=outdir / "variance_separation.png",
    )
    plot_threshold_scan(var_scan, outdir / "variance_threshold_scan.png")
    plot_best_accuracy(var_best, outdir / "variance_best_accuracy.png")

    # mad05 classifier
    mad_scan = threshold_scan(
        batch_df,
        stat_name="mad05",
        thresholds=np.linspace(0.0, 0.5, 51),
        predict_z_if_above=True,
    )
    mad_best = best_thresholds(mad_scan)

    mad_scan.to_csv(outdir / "mad05_threshold_scan.csv", index=False)
    mad_best.to_csv(outdir / "mad05_best_thresholds.csv", index=False)

    plot_stat_separation(
        batch_df,
        stat_name="mad05",
        ylabel=r"Mean $|f-1/2|$",
        outpath=outdir / "mad05_separation.png",
    )
    plot_threshold_scan(mad_scan, outdir / "mad05_threshold_scan.png")

    print("\nPair-label means:")
    print(df.groupby(["family", "pair_label"])["forbidden_rate"].agg(["mean", "std"]))

    print("\nBest variance thresholds:")
    print(var_best[["n_unknowns", "threshold", "acc_Z", "acc_X", "overall_accuracy"]])

    print(f"\nSaved results in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
