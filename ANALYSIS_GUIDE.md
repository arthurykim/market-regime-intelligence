# What We Built and How to Think About It

## The Purpose

This project answers one question: **What kind of market environment are we in right now?**

Not "what will the market do tomorrow" — that's prediction, and it's a different (much harder) problem. This is classification. We look at observable data — how volatile prices have been, how far below recent highs we are, what the VIX is saying — and we label the current environment.

The three labels are intentionally simple:
- **calm**: Volatility is low, prices are near highs, the options market isn't pricing in fear. This is the default state. The market spends about 62% of its time here.
- **elevated_risk**: Something is off. VIX is creeping up past 20, or the market has pulled back 5%+. Not a crisis, but not comfortable either. About 25% of trading days fall here.
- **crisis**: VIX is above 30 or the market is down 10%+ from recent highs. Think 2008, COVID March 2020, the April 2025 tariff shock. About 12% of days.

## What Each Feature Tells You

### Log Returns
Daily percentage change in SPY, computed as log(today / yesterday). We use log returns because they're additive over time and have nicer statistical properties than simple returns. On most days, this number is between -2% and +2%.

**What to look at**: Large negative returns (below -3%) tend to cluster. Volatility begets volatility. If you see a string of large negative returns, the regime is likely shifting.

### Realized Volatility (20-day rolling, annualized)
Standard deviation of the last 20 daily returns, scaled up to annual terms by multiplying by sqrt(252). This tells you how choppy the market has actually been recently.

**What to look at**: Realized vol below 10% is very calm (2017 was famously low-vol). Above 20% is elevated. Above 40% is extreme (COVID peak hit ~80%). Compare this to VIX — when VIX is much higher than realized vol, the market is pricing in fear of something that hasn't happened yet.

### Rolling Drawdown (60-day)
How far below the 60-day rolling high the market currently sits. A value of 0 means SPY is at or near its recent peak. A value of -0.15 means it's 15% below the 60-day high.

**What to look at**: Drawdowns below -5% are uncomfortable. Below -10% is a correction. Below -20% is a bear market. The 60-day window captures medium-term damage — it smooths out single-day drops but still reacts to multi-week selloffs.

### Return Z-Score (20-day)
How unusual today's return is relative to the last 20 days. A z-score of +2 means today's return was 2 standard deviations above the recent average. This helps detect days where something sudden happened.

**What to look at**: Z-scores beyond +/- 2 are interesting. Beyond +/- 3 is rare. Large negative z-scores often mark the start of a regime shift. Large positive z-scores during a crisis period can signal a reversal (or a dead cat bounce — the feature doesn't know which).

### VIX-Realized Vol Spread
VIX (forward-looking implied volatility from options) minus actual realized volatility. When this spread is large and positive, the market is scared of something that hasn't shown up in prices yet. When it's negative, prices are more volatile than the options market expected.

**What to look at**: A persistently high spread (VIX >> realized vol) suggests the market is on edge. A collapsing spread during high-vol periods means realized is catching up to what VIX already priced in. During the April 2025 tariff shock, VIX spiked to 52 — far above what realized vol showed at the time.

## How the Regime Classifier Works

The classifier is deliberately simple. It uses two inputs: VIX level and drawdown depth.

```
if VIX >= 30 or drawdown <= -10%:  → crisis
elif VIX >= 20 or drawdown <= -5%: → elevated_risk
else:                              → calm
```

Crisis takes priority over elevated_risk. The thresholds (VIX 20/30, drawdown -5%/-10%) are standard conventions in financial risk management, not parameters tuned to our data. This is a feature, not a bug — it means the classifier generalizes to new market environments without overfitting.

## Directions for Further Analysis

### Things you can explore right now with the existing code:

1. **Regime transition analysis**: How often does the market go directly from calm to crisis vs. transitioning through elevated_risk first? Export the regimes and look at transition frequencies. This tells you whether risk escalates gradually or hits suddenly.

2. **Duration analysis**: How long does the average crisis last? How long does calm last? Group consecutive days of the same regime and look at the distribution of durations. Crisis regimes that last 3 days feel very different from ones that last 60.

3. **Feature behavior by regime**: Compute summary stats (mean, median, std) of each feature grouped by regime. For example: what's the average daily return during crisis vs. calm? What's the average VIX-realized vol spread in each regime?

4. **Historical event mapping**: Overlay major events (Lehman collapse, COVID lockdown, SVB failure, April 2025 tariffs) onto the regime timeline. See how quickly the classifier reacts and how long it stays in crisis.

5. **Threshold sensitivity**: Change the VIX thresholds in config.py (try 18/28 or 22/32) and see how the regime distribution shifts. This helps you understand how sensitive the classification is to parameter choices.

### Future extensions worth considering:

6. **Hidden Markov Model**: Instead of fixed thresholds, let the data learn regime boundaries. A 3-state HMM on returns and VIX would discover regimes statistically. Compare its output to the rule-based classifier.

7. **Regime-conditional return distributions**: Fit different probability distributions to returns within each regime. Calm-period returns may be approximately normal. Crisis-period returns are likely fat-tailed and negatively skewed. This is useful for risk modeling.

8. **Sector-level analysis**: Download sector ETFs (XLF, XLK, XLE) and see if they enter/exit regimes at different times. Financials might lead into crisis; tech might lag.

9. **Real-time alerting**: Add a check that runs daily, compares today's regime to yesterday's, and sends a notification on regime transitions. This is a natural extension of the API.

10. **Backtesting framework**: Test a simple hypothesis: "Does reducing equity exposure during elevated_risk and crisis regimes improve risk-adjusted returns?" This is not a trading strategy — it's a statistical test of whether the regime labels carry information.

## What This Project Demonstrates

For a portfolio or interview context, this project shows:

- **Data engineering**: Ingesting, caching, cleaning, and aligning multi-source financial data
- **Feature engineering**: Computing interpretable risk metrics from raw prices
- **Statistical thinking**: Choosing features and thresholds based on domain knowledge, not just model fitting
- **Software architecture**: Modular design with clear separation of concerns (ingestion → features → classification → API)
- **API design**: Clean REST endpoints with Pydantic validation and typed responses
- **Testing**: 31 tests covering edge cases, boundary conditions, and integration
- **Visualization**: Publication-quality charts that communicate findings to non-technical stakeholders

The key thing to convey in any interview: you understood the problem before you wrote code. You chose simple, interpretable methods because the goal is descriptive analysis, not prediction theater. You wrote tests because reliability matters. You exposed the results through an API because data that lives in a notebook is data that doesn't get used.
