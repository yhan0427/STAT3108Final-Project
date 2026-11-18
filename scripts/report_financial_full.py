"""Publish matched information controls and conditional uncertainty, including raw losses."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / "src"))
from volrisk.io import read_json


def main():
    primary = pd.read_csv(ROOT / "results/FINANCIAL_FULL_V1.csv")
    broad = pd.read_csv(ROOT / "results/RIDGE_REGULARIZATION_AUDIT_V1.csv")
    audit = read_json(ROOT / "results/FINANCIAL_FULL_AUDIT_V1.json")
    ridge = read_json(ROOT / "results/RIDGE_REGULARIZATION_AUDIT_V1.json")
    dataset = read_json(ROOT / "results/FINANCIAL_FULL_DATA_V1.json")
    lines = ["# Matched financial information experiment", "",
        "Question: do financial values add information beyond the same market history, financial-field "
        "availability and filing age? This is an exploratory development study, not a fresh temporal test.", "",
        f"Data: {dataset['retained_non_financial_firms']} retained non-financial reference-cohort companies; "
        "327 fitting firms / 550,253 origins; 327 non-held validation firms / 155,750 common-context origins; "
        "81 held-role firms / 38,961 evaluation origins in 2024–2025. Missing financial values do not remove "
        "market observations. Company roles, execution-aligned target and training-only preprocessing are unchanged.", "",
        "## Information controls", "",
        "Every LSTM has the same sixteen-coordinate financial branch, architecture, three seeds and epoch budget. "
        "Market-only zeros every branch input. Availability retains filing age and eight missing flags while zeroing "
        "the seven economic values. Financial uses all values plus those controls. Ridge receives the same sixty-day "
        "market lag and current static coordinates. Each model has its own validation-only scalar calibration. "
        "The V3 quantities are alternatives with explicit economic definitions, not imputed total liabilities or operating margins.", "",
        "## Primary frozen run", "", "| Model | MAE (pp) | RMSE (pp) | QLIKE |",
        "|---|---:|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {row.model} | {row.mae*100:.3f} | {row.rmse*100:.3f} | {row.qlike:.6f} |")
    lines += ["", "## Financial content beyond availability", "",
        "| Backbone / regularization | Loss difference: financial − availability | 20-date 95% interval | 60-date 95% interval |",
        "|---|---:|---|---|"]
    sources = [("LSTM primary", audit["paired_intervals"], "LSTM"),
               ("Ridge primary grid", audit["paired_intervals"], "Ridge"),
               ("Ridge broad grid", ridge["populations"]["non_financial"]["paired_intervals"], "Ridge")]
    for label, intervals, backbone in sources:
        key = f"{backbone}_financial_calibrated-minus-{backbone}_availability_calibrated"
        entry = intervals[key]
        a, b = entry["20"], entry["60"]
        lines.append(f"| {label} | {a['mean_difference']:.6f} | [{a['ci95_low']:.6f}, {a['ci95_high']:.6f}] | "
                     f"[{b['ci95_low']:.6f}, {b['ci95_high']:.6f}] |")
    lines += ["", "Negative differences favor financial values. All financial-versus-availability intervals above "
        "cross zero. Calibrated LSTM QLIKE changes from 0.303148 (market) to 0.268457 (availability) to "
        "0.261979 (financial). Most of this observed market-to-financial difference is also obtained by the "
        "availability/age control; this is not a causal percentage decomposition. The additional financial-content "
        "effect is uncertain, not proved absent.", "",
        "Raw LSTM QLIKE is 0.341923 for availability versus 0.361475 for financial values: the ranking reverses "
        "after calibration. Calibrated financial MAE is lower in both development years, but two already-inspected "
        "years do not establish stable out-of-period improvement. Retain raw/calibrated metrics and all company/year outcomes.", "",
        "## Stronger regularization audit", "",
        "The primary Ridge grid selected its maximum alpha. A separate, explicitly exploratory broad-grid audit "
        "uses [0.01, 1, 100, 10000, 100000, 1000000, 10000000, 100000000] with validation log-MSE selection. "
        "All optima are now interior. Earlier experiments and scores are preserved.", "",
        "| Population | Model | Selected alpha | MAE (pp) | QLIKE |", "|---|---|---:|---:|---:|"]
    for row in broad.loc[broad.model.str.endswith("_calibrated")].itertuples():
        lines.append(f"| {row.population} | {row.model} | {row.selected_alpha:g} | {row.mae*100:.3f} | {row.qlike:.6f} |")
    lines += ["", "The full-market audit is a different population (97 evaluated firms) from the non-financial "
        "information experiment (81 firms). Do not compare their absolute losses as an information treatment. "
        "Expanded Ridge regularization does not establish a financial-content advantage over availability: its "
        "incremental intervals still cross zero. The earlier LSTM/Ridge evidence should be interpreted alongside "
        "this stronger-baseline audit rather than silently replacing the recorded historical grid.", "",
        "## Interpretation and next decision", "",
        "This motivates a separately registered age-versus-missingness diagnostic. Reporting-calendar information, "
        "taxonomy/industry differences and coverage can all influence the control; no one mechanism is established "
        "by the combined availability branch. Text integration must include analogous availability, age and length "
        "controls rather than interpreting every multimodal gain as semantic information.", "",
        "Date-block intervals are conditional on fitted ensembles and the observed reference cohort. "
        "They do not capture all model-selection, company-population or data-source uncertainty, and no multiplicity "
        "adjustment or confirmatory claim is made. Original-filing spot checks are bounded; entity transitions, "
        "custom tags and the current SEC database's historical limitations remain. No portfolio benefit or alpha "
        "follows from these forecast losses alone. The 2026 market holdout remains sealed.", ""]
    (ROOT / "results/FINANCIAL_FULL_V1.md").write_text("\n".join(lines))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = {"LSTM": "#183b62", "Ridge": "#498bb2"}
    groups = ["market", "availability", "financial"]
    positions = np.arange(3)
    for offset, backbone in [(-.18, "LSTM"), (.18, "Ridge")]:
        data = primary if backbone == "LSTM" else broad.loc[broad.population.eq("non_financial")]
        values = [float(data.loc[data.model.eq(f"{backbone}_{g}_calibrated"), "qlike"].iloc[0]) for g in groups]
        bars = left.bar(positions + offset, values, width=.34, color=colors[backbone], label=backbone + (" (broad grid)" if backbone == "Ridge" else ""))
        for bar, value in zip(bars, values):
            left.text(bar.get_x() + bar.get_width()/2, value + .005, f"{value:.3f}", ha="center", fontsize=8)
    left.set_xticks(positions, ["Market", "+ availability\n& age", "+ financial\nvalues"])
    left.set_ylim(0, .36)
    left.set_ylabel("Calibrated QLIKE (lower is better)")
    left.set_title("Matched information sets", fontsize=11)
    left.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    right.axvline(0, color="#8c98a1", linestyle="--", linewidth=1)
    labels, ypositions = [], []
    for i, (backbone, intervals) in enumerate([
        ("LSTM", audit["paired_intervals"]),
        ("Ridge", ridge["populations"]["non_financial"]["paired_intervals"])]):
        key = f"{backbone}_financial_calibrated-minus-{backbone}_availability_calibrated"
        for j, block in enumerate(["20", "60"]):
            yposition = 3 - (i*2+j)
            point = intervals[key][block]
            mean = point["mean_difference"]
            right.errorbar(mean, yposition, xerr=[[mean-point["ci95_low"]], [point["ci95_high"]-mean]],
                           fmt="o", capsize=3, color=colors[backbone], markersize=5)
            ypositions.append(yposition)
            labels.append(f"{backbone}, {block}-date blocks")
    right.set_yticks(ypositions, labels, fontsize=8)
    right.set_xlabel("QLIKE difference: financial − availability")
    right.set_xticks([-.015, -.010, -.005, 0.])
    right.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    right.set_title("Increment beyond disclosure metadata", fontsize=11)
    right.set_ylim(-.6, 3.6)
    for ax in [left, right]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if ax is left else "x", color="#e7edf1", linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Financial values versus disclosure-state controls", fontsize=13, fontweight="bold")
    fig.text(.5, .015, "81 held-role companies · inspected 2024–2025 development period · conditional 95% date-block intervals · 2026 sealed", ha="center", fontsize=8, color="#53616c")
    fig.tight_layout(rect=[0, .05, 1, .95])
    fig.savefig(ROOT / "results/FINANCIAL_FULL_V1.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
