# Bitcoin Sentiment Forecasting — completed historical experiment

Completed 7 October 2026. This report describes the local research prototype and its saved results.

## Research question and result

Can an LSTM improve next-day Bitcoin price forecasts beyond predicting that tomorrow's price equals today's? Does adding Fear and Greed or recent volatility help?

The tested models did not demonstrate a consistent advantage. In the final reserved period, persistence achieved the lowest mean absolute error (MAE), our primary metric. This is evidence about this implementation and dataset; it does not establish that all LSTMs or all sentiment signals are ineffective.

| Final forecast | MAE (USD) | RMSE (USD) | MAE improvement vs persistence |
|---|---:|---:|---:|
| Persistence | 1,488.92 | 2,062.49 | 0.00% |
| Price-only LSTM | 1,501.59 | 2,062.36 | -0.85% |
| Price + volatility LSTM | 1,514.03 | 2,074.88 | -1.69% |

Final target dates: 27 February 2025–23 May 2026; 451 observations. LSTM forecasts are the arithmetic mean of price forecasts from seeds 1, 42, and 123. No best seed was selected. The price-only model's near-identical RMSE does not change the primary MAE conclusion. No statistical significance or trading-profit claim is made.

## What was built

The local Streamlit dashboard provides historical reference prices, baseline and regression results, LSTM comparisons, three earlier-period comparisons, a volatility experiment, error inspection, a news-processing summary, and the final evaluation.

The price and news research tracks are separate. CryptoPulse's code processed real news, but those news scores were not used to train or evaluate a forecasting model.

## Data and prediction timing

Coin Metrics' official community archive provides Bitcoin `PriceUSD`, a USD reference price at UTC day end. It is an aggregated reference price, not a directly executable exchange quote. The usable series spans 18 July 2010–23 May 2026 and was already stale when this project began.

Alternative.me's downloaded Fear and Greed archive contains 3,167 observations from 1 February 2018 through 7 October 2026, with four missing calendar days. Experiments used complete required windows without filling gaps from future observations.

There were 3,001 eligible examples in the initial ten-day window dataset. All historical model comparisons use matching eligible dates. A ten-day sequence of daily log returns supplies price features. Models predict next-day log return, then convert it to a USD price using the known origin price. This return target differs from the reference notebook's direct scaled-price target.

Fear and Greed features use a one-calendar-day lag. Exact historical publication/processing availability and provider revisions were not independently verified; the lag is an assumption, not proof of full point-in-time integrity. The volatility feature is the population standard deviation of the last three daily log returns at each sequence step and uses no later prices.

## Evaluation design

Initial development: 2,099 training examples and 449 validation examples, with a boundary gap. Training-only statistics scale features and returns.

Earlier development checks used expanding training histories followed by January–June 2021, July–December 2021, and January–June 2022. Each LSTM was freshly trained for seeds 1, 42, and 123. These periods were selected after the initial validation result was inspected; they are development diagnostics, not independent final evidence.

| Development period | Price-only MAE improvement | Fear and Greed MAE improvement | Volatility MAE improvement |
|---|---:|---:|---:|
| Jan–Jun 2021 | +1.15% | -0.49% | +1.13% |
| Jul–Dec 2021 | -0.67% | -0.32% | -0.55% |
| Jan–Jun 2022 | +0.14% | -0.11% | +0.41% |

These development values average per-seed errors; final metrics score the averaged price forecasts. They are different aggregation methods and should not be equated.

After development, the final settings were recorded before computing holdout forecasts: one 16-unit LSTM, one scalar dense output, Adam at 0.001, 20 epochs, batch size 32, no shuffling, ten input steps, and three fixed seeds. Models were retrained on 2,549 earlier examples, whose last target was 25 February 2025. The first final forecast origin was 26 February, targeting 27 February.

The final comparison included persistence, price-only LSTM, and price-plus-volatility LSTM. Fear and Greed was not included in the final comparison because it showed no consistent development benefit. Consequently, the final period does not test sentiment's forecasting value directly.

The final period is now inspected. Its outcomes must not become a tuning target presented as an untouched test. Future changes require genuinely new evaluation data and a new protocol.

## Error analysis

Models predicted small changes relative to actual daily movements. In Jan–Jun 2021, the price-only model's mean absolute predicted move was about 0.38%, versus 3.58% actually observed. On 8 February 2021, seed 1 predicted about +0.43% versus an actual +18.25% move.

Small predictions are not inherently wrong: uncertain returns can have conditional means near zero. Increasing their magnitude without predictive evidence may worsen accuracy. Adding volatility was a controlled development experiment, not a guaranteed remedy. Its limited development gains did not carry into the final MAE comparison.

## Connection to previous projects

- **CryptoPulse AI:** reused its cleaning, Bitcoin evidence resolution, duplicate grouping and VADER scoring code. A separate private news collection exercised ingestion, scoring and human review. No real-data sentiment accuracy estimate was established. Provider text and private news-derived processing statistics are excluded from this public report.
- **SignalGuard:** adapted its evaluation and failure-analysis approach. Its stock-direction model and Apple dataset were not imported into Bitcoin training. No strategy simulation was performed here.
- **Reference notebook:** supplied the learning idea of sequence forecasting. The new experiment corrected training/test preprocessing and date-boundary handling, added baselines and a reserved final evaluation, and changed the prediction target to returns.

News publication time, actual retrieval time, scoring completion, and signal availability remain distinct. News text and scoring evidence are private local development artifacts. Neither real-news training eligibility nor public-release permission has been established. There is insufficient overlapping news/price history for a news forecasting claim.

## Verification and limitations

Saved final predictions were independently recalculated to reconcile MAE/RMSE for all three forecasts, all 451 dates, and source hashes. This checks result accounting; it does not independently rerun training, establish source accuracy, or verify historical publication/revision timing.

Other limits include one asset, a limited historical archive endpoint, fixed simple architecture, no formal uncertainty analysis, no representative human-labeled sentiment evaluation, no transaction-cost simulation, and no live-price or current-forecast service. The code currently depends on local files under `work/`; the public demo is portable, while exact historical reproduction still requires the original local inputs.

## Reopen the dashboard

From the workspace root in VS Code:

```powershell
python -m streamlit run scripts/bitcoin_page.py
```

The final evaluator detects an existing result and avoids silently retraining it. Preserve `work/final_evaluation/` with its frozen plan, result, predictions and verification. Provider data in `work/` must not be committed as an unrestricted dataset.

## Sources and attribution

Price data: Coin Metrics, Inc., https://github.com/coinmetrics/data , licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Transformations include selecting the USD price, constructing lagged sequences, training forecasts and computing errors. Definition: https://github.com/coinmetrics/docs-website/blob/master/asset-metrics/market/priceusd.md .

Fear and Greed: https://alternative.me/crypto/fear-and-greed-index/ . The index was normalized and lagged; source attribution is required.

CryptoPulse: https://github.com/KaviyaVelmurugan/cryptopulse-ai . SignalGuard: https://github.com/KaviyaVelmurugan/SignalGuard . Reference: https://github.com/silvainfm/FinTech-Projects/blob/main/Stock%20Predictor/lstm_stock_predictor_fng.ipynb .

News development access: https://newsapi.org/terms and https://newsapi.org/pricing . No article text or provider news dataset is included in this report. Public release of the complete news-derived product remains a separate rights review.
