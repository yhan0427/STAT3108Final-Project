# Portfolio Risk Lab: application protocol v1

## Research question and scope

Can forecast-based equity exposure improve portfolio risk-target tracking relative to historical volatility, after delayed execution and trading costs? Secondary outcomes are net returns, drawdowns, equity exposure, turnover and costs. This application evaluates a decision rule, not stock-return prediction or arbitrage.

The first release reuses saved V2 forecasts and local adjusted daily prices for thirty fixed non-financial large caps. The sample is survivor-selected and not reconstructed S&P 500 membership. Earlier experiments have already examined these periods. All portfolio scenarios are exploratory, including presets designed after earlier results; a new confirmatory sample remains future work.

## Portfolio risk

The input is a nonnegative vector of relative stock weights summing to one. At signal close t, fit Ledoit-Wolf covariance to the last sixty available daily log returns of the chosen stocks. Convert this estimate to correlation R. Both rules use the same R, conditional on the selected basket.

Historical rule: marginal annual volatility s is the saved trailing twenty-session estimate. Forecast rule: s is the selected model's saved annual volatility forecast. In either case Sigma = diag(s) R diag(s); portfolio volatility is sqrt(w' Sigma w). The Euler variance share for stock i is w_i (Sigma w)_i / (w' Sigma w). Shares sum to one but can be negative. They are risk attribution, not expected losses or return forecasts.

Target equity exposure is min(1, target / estimated portfolio volatility); the remaining fraction is cash. Target risk is an input, not a promise. When the basket is less volatile than the target, the no-leverage constraint prevents reaching the target. Correlations are estimated historically, not forecast. Reference implementation: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html.

## Four comparisons

- Buy and hold: enter the supplied basket once and let relative weights drift.
- Fixed 50% equity: rebalance to half the supplied basket and half cash on the common schedule. This illustrates the effect of lower exposure; it is not a matched-risk benchmark.
- Historical-risk rule: scale the supplied basket using trailing marginal volatilities and historical correlation.
- Forecast-risk rule: use the same rule and correlation estimate with model-predicted marginal volatilities.

The latter three restore the supplied relative stock weights when rebalancing. Between trades, holdings and weights drift with realized returns. The two dynamic rules share risk targets, not necessarily realized volatility or average exposure. A claim of predictive timing benefit would need stronger exposure-matched attribution than this first release supplies.

## Timing and accounting

Take every twentieth saved signal date, anchored to the first date of each selected protocol. Information is available after the signal close; execution occurs at the next trading close. The account receives existing holdings' return first and then executes. New weights earn returns only from the following session. Initial capital starts in cash at the first execution date; it pays entry costs. The end date is the latest saved target end in the selected protocol, with positions marked to market and no forced liquidation.

The original forecasting label covers the next twenty close-to-close returns. The next-close execution shifts the investable window by a session. This is an explicit implementation-lag experiment; do not label the resulting portfolio risk estimate an exactly aligned forecast of tradable twenty-session variance.

Returns use changes in adjusted closes as a vendor total-return proxy; do not add distributions again. Prices must be positive and calendars complete. Historical vendor revisions and adjustment factors are limitations. Cash return is zero. There is no leverage, shorting, tax model, liquidity/market-impact model or broker connection.

Proportional cost c applies to the absolute dollar amount traded in stocks, counting purchases and sales. For pre-trade NAV V, holdings h and desired post-cost weights u, solve C = c sum_i |u_i (V-C) - h_i|. Set holdings to u(V-C) and cash to (1-sum u)(V-C). This preserves self-financing even at 100% equity. Report accumulated cash costs in units of initial capital and gross dollars traded divided by pre-trade NAV as turnover (no half-turnover convention).

## Evaluation

The declared grid in config/portfolio-v1.json covers two fixed baskets, four protocols, Ridge and market LSTM, and costs of 0/10/25 bps. Target annual volatility is 10%. The main comparison uses 10 bps; all scenarios are retained. UI users may explore other legal weights, targets and costs, but those explorations are not new predeclared tests.

Primary metric: mean absolute gap between trailing twenty-session annualized standard deviation of net strategy returns and the target. Also report compounded net return, annualized return, annualized net volatility, maximum drawdown including initial costs, average equity fraction, gross turnover and cash trading costs. A supplemental Sharpe uses a zero risk-free rate to match the cash assumption; it is not a Treasury-excess-return Sharpe. Initial entry costs belong to the path. The full-period 2024-25 and annual 2024 tests overlap.

Conditional paired date-block intervals are diagnostics. Neither numerous scenarios nor overlapping return windows constitute independent replications. Lower drawdown or volatility may reflect more cash; more return may reflect more equity. No current result demonstrates universal superiority, alpha, actual user savings or deployment readiness.

## Application and AI boundary

POST /api/portfolio/evaluate accepts protocol, model, date, weights, target, cost_bps and reveal. Weights are decimal fractions totaling one; target is annualized decimal volatility; cost_bps is basis points per traded dollar. Invalid fields fail rather than silently normalize or remove stocks.

The response contains current decision estimates, assumptions and computed evidence. It includes backtest paths and metrics only when reveal=true. The selected date controls the decision view; revealed results cover the entire selected protocol, explicitly including dates before and after that date. POST /api/portfolio/chat reconstructs the same evidence on the server. E7 is the only revealed-hindsight evidence card. Edited inputs invalidate the prior answer. Exports preserve the applied configuration and evidence; credentials never enter exports.

The assistant explains scenarios and limitations. It does not calculate weights, select winners or execute orders. Provider access still needs a local OpenAI API key. No real model round-trip is claimed until tested.

## Next research stage

Choose one recent finance/time-series paper after reviewing its code, target, data and pretraining overlap. Add a reproducible comparator through the existing forecast interface, then register a genuinely new time/company evaluation and cost/cash/correlation sensitivities. Gaps in numerical stability and financial coverage from V2/V3 remain disclosed. Do not change the primary outcome after seeing portfolio returns.

## 2026-11-11 accounting correction and aligned extension

The first V1 CSV computed turnover with previous-close NAV, rather than execution-day pre-trade NAV. The implementation now matches the definition above. `results/PORTFOLIO_V1_TURNOVER_CORRECTION.csv` gives corrected turnover for all 192 records; the companion JSON confirms every other metric is unchanged. The original CSV is retained as an archived result.

A separate execution-aligned TTM / LogHAR-proxy / Ridge comparison is now documented in `docs/ALIGNED_METHOD_TRANSFER.md` and `results/ALIGNED_V1.md`. It does not silently replace the demo's V2 forecasting inputs or establish a new holdout.
