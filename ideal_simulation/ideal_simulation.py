"""
Ideal Qiskit simulation for the no-CNOT graph-state basis-discrimination protocol.

Purpose
-------
This script implements the graph-state protocol in Qiskit and studies the
batch-level variance classifier in the ideal noiseless limit.

Question addressed:
    Given a batch of k independently selected same-basis input pairs,
    how well does the empirical variance of the forbidden-signature rate
    distinguish the Z-basis distribution from the X-basis distribution?

Outputs
-------
Creates:
    ideal_qiskit_batch_variance_results/

with CSV files and plots:
    ideal_pair_forbidden_rates.csv
    ideal_batch_variance_samples.csv
    ideal_threshold_scan.csv
    ideal_best_thresholds.csv
    ideal_pair_forbidden_rates.png
    ideal_variance_separation_vs_k.png
    ideal_accuracy_vs_k.png
    ideal_variance_histograms.png
"""

from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


Z_PAIR_LABELS = ["00", "01", "10", "11"]
X_PAIR_LABELS = ["++", "+-", "-+", "--"]
ALL_PAIR_LABELS = Z_PAIR_LABELS + X_PAIR_LABELS

REFERENCE_SIGNATURES = {(+1, +1, +1), (-1, -1, +1)}
EVEN_SIGNATURES = {
    (+1, +1, +1),
    (+1, -1, -1),
    (-1, +1, -1),
    (-1, -1, +1),
}
FORBIDDEN_SIGNATURES = EVEN_SIGNATURES - REFERENCE_SIGNATURES


def prepare_single(qc: QuantumCircuit, qubit: int, label: str) -> None:
    """Prepare |0>, |1>, |+>, or |->."""
    if label == "0":
        return
    if label == "1":
        qc.x(qubit)
    elif label == "+":
        qc.h(qubit)
    elif label == "-":
        qc.x(qubit)
        qc.h(qubit)
    else:
        raise ValueError(f"Unknown label: {label}")


def build_graph_circuit(pair_label: str) -> QuantumCircuit:
    """
    No-CNOT graph-state circuit.

    Qubit layout:
        q0, q1 = unknown input pair
        q2 = a1
        q3 = a2
        q4 = a3
    """
    qc = QuantumCircuit(5)

    prepare_single(qc, 0, pair_label[0])
    prepare_single(qc, 1, pair_label[1])

    # Ancillas in |+>.
    qc.h(2)
    qc.h(3)
    qc.h(4)

    # Graph-state CZ layer.
    qc.cz(0, 2)
    qc.cz(0, 3)
    qc.cz(1, 3)
    qc.cz(1, 4)

    # X-basis measurement of ancillas = H then computational probabilities.
    qc.h(2)
    qc.h(3)
    qc.h(4)

    return qc


def bit_to_eig(bit: int) -> int:
    return +1 if bit == 0 else -1


def ancilla_signature_from_index(index: int) -> tuple[int, int, int]:
    """
    Qiskit statevector index is little-endian: bit q is qubit q.
    After the final H gates, Z bits of q2,q3,q4 correspond to X outcomes.
    """
    b2 = (index >> 2) & 1
    b3 = (index >> 3) & 1
    b4 = (index >> 4) & 1
    return (bit_to_eig(b2), bit_to_eig(b3), bit_to_eig(b4))


def signature_probabilities(pair_label: str) -> dict[tuple[int, int, int], float]:
    qc = build_graph_circuit(pair_label)
    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2

    sig_probs = {sig: 0.0 for sig in itertools.product([+1, -1], repeat=3)}

    for idx, p in enumerate(probs):
        if p < 1e-15:
            continue
        sig = ancilla_signature_from_index(idx)
        sig_probs[sig] += float(p)

    return {sig: 0.0 if abs(p) < 1e-12 else p for sig, p in sig_probs.items()}


def forbidden_rate(pair_label: str) -> float:
    probs = signature_probabilities(pair_label)
    return float(sum(p for sig, p in probs.items() if sig in FORBIDDEN_SIGNATURES))


def build_pair_table() -> pd.DataFrame:
    rows = []
    for label in ALL_PAIR_LABELS:
        family = "Z" if label in Z_PAIR_LABELS else "X"
        rows.append(
            {
                "pair_label": label,
                "family": family,
                "forbidden_rate": forbidden_rate(label),
            }
        )
    return pd.DataFrame(rows)


def simulate_batch_variances(
    pair_table: pd.DataFrame,
    k_values=tuple(range(2, 21)),
    n_batches: int = 10000,
    seed: int = 2027,
) -> pd.DataFrame:
    """
    Simulate batches of size k, where each batch is drawn entirely from
    either the Z-pair distribution or the X-pair distribution.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for k in k_values:
        for family in ["Z", "X"]:
            sub = pair_table[pair_table["family"] == family].reset_index(drop=True)
            values = sub["forbidden_rate"].to_numpy(dtype=float)
            labels = sub["pair_label"].to_numpy()

            for batch_id in range(n_batches):
                idx = rng.integers(0, len(values), size=k)
                sampled_f = values[idx]
                sampled_labels = labels[idx]

                rows.append(
                    {
                        "family": family,
                        "k_pairs": k,
                        "n_unknowns": 2 * k,
                        "batch_id": batch_id,
                        "mean_f": float(np.mean(sampled_f)),
                        "var_f": float(np.var(sampled_f, ddof=0)),
                        "sampled_labels": " ".join(sampled_labels),
                    }
                )

    return pd.DataFrame(rows)


def threshold_scan(batch_df: pd.DataFrame, thresholds=None) -> pd.DataFrame:
    """
    Classifier:
        predict Z if var_f > threshold,
        otherwise predict X.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.25, 501)

    rows = []

    for k in sorted(batch_df["k_pairs"].unique()):
        sub = batch_df[batch_df["k_pairs"] == k].copy()
        true = sub["family"].to_numpy()

        for t in thresholds:
            pred = np.where(sub["var_f"].to_numpy() > t, "Z", "X")

            acc_z = float(np.mean(pred[true == "Z"] == "Z"))
            acc_x = float(np.mean(pred[true == "X"] == "X"))
            overall = 0.5 * (acc_z + acc_x)

            rows.append(
                {
                    "k_pairs": k,
                    "n_unknowns": 2 * k,
                    "threshold": float(t),
                    "acc_Z": acc_z,
                    "acc_X": acc_x,
                    "overall_accuracy": overall,
                }
            )

    return pd.DataFrame(rows)


def best_thresholds(scan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(scan_df["k_pairs"].unique()):
        sub = scan_df[scan_df["k_pairs"] == k]
        rows.append(sub.iloc[sub["overall_accuracy"].argmax()].to_dict())
    return pd.DataFrame(rows)


def plot_pair_rates(pair_table: pd.DataFrame, outpath: Path) -> None:
    order = Z_PAIR_LABELS + X_PAIR_LABELS
    data = pair_table.set_index("pair_label").loc[order]

    plt.figure(figsize=(7, 4.5))
    plt.bar(range(len(order)), data["forbidden_rate"].to_numpy())
    plt.xticks(range(len(order)), order)
    plt.xlabel("Input-pair label")
    plt.ylabel("Forbidden-signature rate")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.title("Ideal pair-level graph-state response")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def plot_variance_separation(batch_df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(7, 4.8))

    for family in ["Z", "X"]:
        xs, means, stds = [], [], []
        for k in sorted(batch_df["k_pairs"].unique()):
            sub = batch_df[(batch_df["family"] == family) & (batch_df["k_pairs"] == k)]
            xs.append(k)
            means.append(sub["var_f"].mean())
            stds.append(sub["var_f"].std(ddof=1))
        plt.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=f"True {family}")

    plt.xlabel("Batch size k (input pairs)")
    plt.ylabel("Empirical variance of forbidden-signature rate")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Ideal batch-level variance separation")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def plot_accuracy(best_df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(7, 4.8))
    plt.plot(best_df["k_pairs"], best_df["overall_accuracy"], marker="o", label="Overall")
    plt.plot(best_df["k_pairs"], best_df["acc_Z"], marker="o", label="True Z")
    plt.plot(best_df["k_pairs"], best_df["acc_X"], marker="o", label="True X")
    plt.xlabel("Batch size k (input pairs)")
    plt.ylabel("Classification accuracy")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Ideal variance-classifier accuracy")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


def plot_variance_histograms(
    batch_df: pd.DataFrame,
    outpath: Path,
    selected_k=(2, 4, 8, 16),
) -> None:
    ncols = len(selected_k)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.5), sharey=True)

    if ncols == 1:
        axes = [axes]

    bins = np.linspace(0, 0.26, 27)

    for ax, k in zip(axes, selected_k):
        for family in ["Z", "X"]:
            sub = batch_df[(batch_df["family"] == family) & (batch_df["k_pairs"] == k)]
            ax.hist(
                sub["var_f"].to_numpy(),
                bins=bins,
                alpha=0.6,
                density=True,
                label=family,
            )
        ax.set_title(f"k={k}")
        ax.set_xlabel("Batch variance")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Density")
    axes[-1].legend()
    fig.suptitle("Ideal finite-batch variance distributions", y=1.03)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    outdir = Path("ideal_qiskit_batch_variance_results")
    outdir.mkdir(parents=True, exist_ok=True)

    k_values = tuple(range(2, 21))
    n_batches = 10000

    pair_table = build_pair_table()
    pair_table.to_csv(outdir / "ideal_pair_forbidden_rates.csv", index=False)

    batch_df = simulate_batch_variances(
        pair_table,
        k_values=k_values,
        n_batches=n_batches,
        seed=2027,
    )
    batch_df.to_csv(outdir / "ideal_batch_variance_samples.csv", index=False)

    scan_df = threshold_scan(batch_df, thresholds=np.linspace(0.0, 0.25, 501))
    best_df = best_thresholds(scan_df)

    scan_df.to_csv(outdir / "ideal_threshold_scan.csv", index=False)
    best_df.to_csv(outdir / "ideal_best_thresholds.csv", index=False)

    plot_pair_rates(pair_table, outdir / "ideal_pair_forbidden_rates.png")
    plot_variance_separation(batch_df, outdir / "ideal_variance_separation_vs_k.png")
    plot_accuracy(best_df, outdir / "ideal_accuracy_vs_k.png")
    plot_variance_histograms(batch_df, outdir / "ideal_variance_histograms.png")

    print("\\nIdeal pair-level forbidden rates:")
    print(pair_table)

    print("\\nBest thresholds and accuracy:")
    print(best_df[["k_pairs", "threshold", "acc_Z", "acc_X", "overall_accuracy"]])

    print(f"\\nSaved results in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
