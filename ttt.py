#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTT Plot Generator
------------------
Reads a CSV with run results and generates Time-to-Target (TTT) plots for each
(instance, target) pair, overlaying multiple algorithms and (optionally) a
shifted-exponential fit per algorithm (quartile-based estimator).

Required CSV columns:
    instance, target, algorithm, seed, time_sec, hit

- "time_sec": time (in seconds) to FIRST achieve a solution >= target.
- "hit": 1 if reached the target (success), 0 otherwise (e.g., timeout).

Outputs (in --out):
    ttt_{instance}__target_{target}.png
    ttt_points_{instance}__target_{target}__{algorithm}.csv

Example:
    python ttt.py results.csv --out out/ --time-limit 600 --fit \
        --title-prefix "MAX-SC-QBF" --normalize-by-total
"""
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"instance", "target", "algorithm", "seed", "time_sec", "hit"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Expected at least: {sorted(required)}"
        )


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase columns for robustness and keep original names when possible
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "hit" in df:
        df["hit"] = df["hit"].astype(int)
    if "time_sec" in df:
        df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    if "target" in df:
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
    return df


def empirical_points(times_success: np.ndarray,
                     normalize_by_total: bool,
                     total_runs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical TTT points.
    - If normalize_by_total is False (default), follow Aiex et al.:
        sort times, p_i = (i - 1/2)/(n), using only successful runs (n = len(times_success)).
      This yields the "median-unbiased" plotting positions for the empirical CDF.
    - If normalize_by_total is True:
        use p_i = i / total_runs (including misses). This is useful when
        runs are capped (e.g., 600s) and some never hit the target.
    """
    times_sorted = np.sort(times_success)
    n = len(times_sorted)
    if n == 0:
        return times_sorted, np.array([])

    if normalize_by_total:
        p = np.arange(1, n + 1) / float(max(total_runs, 1))
    else:
        # classical plotting position: (i - 1/2) / n, but some literature uses n+1
        # We stick to n here to remain close to the original empirical-CDF positions,
        # while the Q-Q fit below uses n+1 as in the reference implementation.
        p = (np.arange(1, n + 1) - 0.5) / float(n)

    return times_sorted, p


def fit_shifted_exponential(times_success: np.ndarray) -> Tuple[float, float]:
    """
    Estimate parameters (mu, lambda) of the shifted exponential using the
    quartile-based estimator from Aiex-Resende-Ribeiro (robust to outliers).

    Steps (matching the reference implementation):
      - Let n = #successes, np1 = n + 1.
      - Define probabilities p_i = (i - 0.5) / (n + 1). (Used to pick theoretical quantiles.)
      - Lower quartile index:  floor( (n + 1) * 0.25 ) - 1  (0-based)
      - Upper quartile index:  floor( (n + 1) * 0.75 ) - 1
      - z_l, z_u: empirical times at those indices; q_l, q_u: Exp(1) theoretical quantiles
        at the same probabilities; q = -ln(1 - p).
      - lambda_hat = (z_u - z_l) / (q_u - q_l)
      - mu_hat     = z_l - lambda_hat * q_l
    """
    t = np.sort(times_success)
    n = len(t)
    if n < 3:
        raise ValueError("Need at least 3 successful runs for a stable quartile fit.")

    np1 = n + 1
    # The reference Perl code builds probs for i=0..n-1 as (i + 0.5)/(n+1)
    probs = (np.arange(n) + 0.5) / float(np1)

    # Indices like the Perl code: int(np1*0.25) and int(np1*0.75) are 1-based;
    # translate to 0-based Python by subtracting 1 (and clamp to [0, n-1]).
    fq = int(np1 * 0.25) - 1
    tq = int(np1 * 0.75) - 1
    fq = max(0, min(fq, n - 1))
    tq = max(0, min(tq, n - 1))

    z_l, z_u = t[fq], t[tq]
    q_l = -np.log(1.0 - probs[fq])
    q_u = -np.log(1.0 - probs[tq])

    # Guard against numerical issues
    if not np.isfinite(q_u - q_l) or (q_u - q_l) == 0.0:
        raise ValueError("Degenerate quartiles for exponential fit.")

    lam = (z_u - z_l) / (q_u - q_l)
    mu = z_l - lam * q_l
    return float(mu), float(lam)


def _nice_ax(ax: plt.Axes) -> None:
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time to target (seconds)")
    ax.set_ylabel("cumulative probability of success")


def plot_pair(df_pair: pd.DataFrame,
              outdir: Path,
              title_prefix: str = "",
              fit: bool = False,
              time_limit: float | None = None,
              normalize_by_total: bool = False,
              min_success_warn: int = 10) -> Path:
    """
    Build a single TTT plot for one (instance, target) pair, overlaying algorithms.
    Returns path to the saved PNG.
    """
    (inst,), (tgt,) = df_pair["instance"].unique(), df_pair["target"].unique()
    total_runs_by_alg = df_pair.groupby("algorithm")["seed"].nunique()

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)

    legend_entries: List[str] = []
    any_curve = False
    fit_rows = []  # to be saved per algorithm

    for alg, g in df_pair.groupby("algorithm"):
        total_runs = int(total_runs_by_alg.get(alg, len(g)))
        succ = g[g["hit"] == 1].sort_values("time_sec")
        times = succ["time_sec"].to_numpy(dtype=float)

        if len(times) == 0:
            # No successes: draw nothing; add a legend note
            legend_entries.append(f"{alg}: 0/{total_runs} hits")
            continue

        times_emp, p_emp = empirical_points(
            times_success=times,
            normalize_by_total=normalize_by_total,
            total_runs=total_runs,
        )
        ax.step(times_emp, p_emp, where="post", linewidth=1.75, alpha=0.95, label=f"{alg} (emp.)")
        any_curve = True

        # Fit (optional)
        mu, lam = (np.nan, np.nan)
        if fit:
            try:
                mu, lam = fit_shifted_exponential(times)
                # Build theoretical curve on a reasonable grid
                tmin = 0.0
                tmax = max(times.max(), time_limit or times.max())
                xs = np.linspace(tmin, tmax, 400)
                # F(t) = 1 - exp(-(t - mu)/lam) for t >= mu; for t < mu, it's < 0 => clamp to 0
                Ft = 1.0 - np.exp(-(xs - mu) / lam)
                Ft[xs < mu] = 0.0
                ax.plot(xs, Ft, linestyle="--", linewidth=1.25, label=f"{alg} (fit)")
            except Exception as e:
                # Skip fit if unstable
                legend_entries.append(f"{alg}: fit skipped ({e})")

        succ_rate = len(times) / max(total_runs, 1)
        if len(times) < min_success_warn:
            legend_entries.append(f"{alg}: {len(times)}/{total_runs} hits (low sample)")
        else:
            legend_entries.append(f"{alg}: {len(times)}/{total_runs} hits")

        # Save per-algorithm points and fit
        out_points = outdir / f"ttt_points_{inst}__target_{tgt}__{alg}.csv"
        pd.DataFrame({
            "instance": inst,
            "target": tgt,
            "algorithm": alg,
            "t": times_emp,
            "p": p_emp,
            "mu_hat": mu,
            "lambda_hat": lam,
            "total_runs": total_runs,
            "successes": len(times),
            "success_rate": succ_rate,
            "time_limit": time_limit,
            "normalize_by_total": normalize_by_total,
        }).to_csv(out_points, index=False)

        fit_rows.append({
            "instance": inst, "target": tgt, "algorithm": alg,
            "mu_hat": mu, "lambda_hat": lam,
            "successes": len(times), "total_runs": total_runs
        })

    if not any_curve:
        plt.close(fig)
        raise RuntimeError(f"No algorithms had successful runs for pair (instance={inst}, target={tgt}).")

    title = f"{title_prefix} — TTT for (inst={inst}, target={tgt})" if title_prefix else f"TTT for (inst={inst}, target={tgt})"
    ax.set_title(title)
    _nice_ax(ax)

    # If a time limit is provided, draw a vertical marker
    if time_limit is not None:
        ax.axvline(time_limit, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.text(time_limit, 0.02, f"limit={time_limit:.0f}s", rotation=90, va="bottom", ha="right", fontsize=8)

    # Add legend notes summarizing hits
    # Burn-in the base legend first to ensure step/fit labels appear
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    if legend_entries:
        # Add a second textbox legend
        ax.text(
            0.02, 0.98,
            "\n".join(legend_entries),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.4"),
        )

    out_png = outdir / f"ttt_{inst}__target_{tgt}.png"
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Save a small per-pair summary for convenience
    pd.DataFrame(fit_rows).to_csv(outdir / f"ttt_fit_summary_{inst}__target_{tgt}.csv", index=False)

    return out_png


def main():
    ap = argparse.ArgumentParser(description="Generate TTT plots per (instance, target).")
    ap.add_argument("csv", type=str, help="Input CSV with columns: instance,target,algorithm,seed,time_sec,hit")
    ap.add_argument("--out", type=str, default="ttt_out", help="Output directory (default: ttt_out)")
    ap.add_argument("--title-prefix", type=str, default="", help="Prefix for figure titles (e.g., 'MAX-SC-QBF')")
    ap.add_argument("--time-limit", type=float, default=None, help="Optional time limit (seconds). Drawn as vertical line.")
    ap.add_argument("--fit", action="store_true", help="Overlay shifted-exponential fit per algorithm (quartile estimator).")
    ap.add_argument("--normalize-by-total", action="store_true",
                    help="Normalize empirical p by total runs (including misses). Default uses only successes per Aiex et al.")
    ap.add_argument("--min-success-warn", type=int, default=10, help="Warn in legend if successes < this threshold (default: 10).")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    _validate_columns(df)
    df = _standardize_columns(df).dropna(subset=["instance", "target", "algorithm", "seed", "time_sec", "hit"])

    # Ensure types are consistent
    df["instance"] = df["instance"].astype(str)
    df["algorithm"] = df["algorithm"].astype(str)

    # Iterate per (instance, target)
    n_pairs = 0
    for (inst, tgt), g in df.groupby(["instance", "target"]):
        try:
            out_png = plot_pair(
                df_pair=g,
                outdir=outdir,
                title_prefix=args.title_prefix,
                fit=args.fit,
                time_limit=args.time_limit,
                normalize_by_total=args.normalize_by_total,
                min_success_warn=args.min_success_warn,
            )
            print(f"[OK] Saved: {out_png}")
            n_pairs += 1
        except Exception as e:
            print(f"[WARN] Skipped pair (inst={inst}, target={tgt}): {e}")

    print(f"Done. Plots generated for {n_pairs} (instance, target) pairs. Output dir: {outdir.resolve()}")


if __name__ == "__main__":
    main()
