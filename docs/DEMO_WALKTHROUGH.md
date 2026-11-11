# Portfolio Risk Lab: live demonstration

Start `.venv/bin/python -m volrisk.server --port 8766`, then open http://127.0.0.1:8766/.

1. Introduce the decision: a portfolio holder wants to know whether forecast-driven risk control improves on a simple historical rule.
2. Use Diversified five, the static 2024-25 protocol, the first decision date, Sequence Ridge, a 10% target and 10 bps cost. Apply the portfolio.
3. Explain correlation-aware portfolio risk and the historical/forecast allocation bars. These are scenario rules, not live trade recommendations.
4. Reveal the full-period historical results. Read risk-target error, return and equity exposure together. A more profitable curve may simply hold more equities. Show an unfavorable case from the completed results as well.
5. Change to Technology five or the 2020 stress window, apply and compare. These are interactive diagnostics, not independent confirmatory tests.
6. Optionally open Research Agent. Without a key, show deterministic evidence and export; do not present a simulated model reply. With a configured key, explain the scope of evidence sent to the provider.
7. Close with the limitations: survivor sample, already-inspected periods, zero cash return, execution lag, fixed correlation methodology and the upcoming recent-method/new-holdout evaluation.

The old individual-stock screening workflow and poster belong to preliminary forecasting experiments. The current application evaluates portfolio decisions; it does not claim the final study or modern-model comparison is complete.
