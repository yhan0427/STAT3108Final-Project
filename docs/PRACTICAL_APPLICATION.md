# Application A: Portfolio Risk Lab

A holder of a stock portfolio wants to understand risk concentration and compare rules for changing equity exposure. The product converts the earlier forecasting work into a decision experiment: input holdings, examine risk, compare stock/cash allocations and reveal cost-adjusted historical consequences.

## User flow

1. Choose a basket or edit holdings and weights, an evaluation period and decision date.
2. Set a risk target and proportional cost assumption; choose an existing forecast model.
3. Apply the experiment. Inspect historical and predicted portfolio volatility, stock/cash exposure and correlation-aware risk contributions.
4. Reveal full-period outcomes and compare buy-and-hold, fixed half-equity, historical-risk and forecast-risk rules.
5. Ask Research Agent to explain the computed evidence or export a reproducible experiment brief.

The homepage uses the established navy/blue gradient, glass panels and metallic accent. Research Agent and About are separate tabs; the homepage has an optional floating assistant. Calculations, charts and export work without AI credentials.

## Contribution and evidence

The intended contribution is an auditable connection between forecasting and portfolio decisions. Forecast accuracy alone is insufficient: evaluate risk-target tracking, returns, costs, turnover and exposure together. The simulator accounts for correlation, lagged execution, initial and subsequent costs, weight drift and self-financing. Its assumptions are in PORTFOLIO_PROTOCOL.md.

Current implementation uses saved Ridge, boosting and LSTM forecasts. It is a functioning retrospective prototype, not evidence that a recent model or financial-text extension improves investment outcomes. V1-V3 forecast experiments remain preliminary supporting evidence. The portfolio report records favorable and unfavorable cases.

## AI role and limitations

Server-generated evidence carries the selected portfolio, risk estimates, allocations and assumptions. Future outcomes and full-period scores are absent unless the user reveals them. The LLM explains this evidence, and must cite valid evidence IDs. Exact numbers remain server-computed; numeric AI prose is rejected. Citation membership checks do not prove semantic accuracy.

API credentials are optional, live only in local-server memory and are cleared on restart. The server binds to loopback; this is not a public deployment authentication design. Missing keys never produce fake AI answers. Changing inputs clears the previous context; no conversation history transfers revealed outcomes into a new decision view.

## Separate evaluations

Portfolio effectiveness: compare defined rules under identical timing and costs, interpret risk and returns alongside cash exposure, test sensitivities and eventually a new holdout. Current scenarios are exploratory and survivor-selected.

Assistant effectiveness: verify factual support, clarity and handling of missing information; optionally compare dashboard-only and assisted interpretation in a consented user study. No user-efficiency gain, semantic accuracy rate, live trading reliability or investment-performance benefit is established by adding a chatbot.
