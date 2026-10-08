# Bitcoin Sentiment Forecasting

A Bitcoin forecasting research dashboard comparing LSTM models against a simple baseline: tomorrow’s price equals today’s price.

The project combines forecasting experiments, sentiment processing, and analysis of prediction errors. A synthetic public demo runs without API keys or private research files.

## Research phases

1. Reviewed the reference Bitcoin prediction notebook and its limitations.
2. Prepared historical Bitcoin prices and Fear and Greed data.
3. Compared persistence, price-only LSTM, and price-plus-sentiment LSTM.
4. Evaluated three earlier periods using three random seeds per LSTM.
5. Investigated forecast errors and added a recent-volatility feature.
6. Evaluated the selected models on a reserved final period.
7. Built the dashboard and prepared a synthetic public demo.

## Final results

| Model                    | MAE (USD) | RMSE (USD) | MAE improvement vs baseline |
|--------------------------|-----------|------------|-----------------------------|
| Persistence              | 1,488.92  | 2,062.49   | 0.00%                       |
| Price-only LSTM          | 1,501.59  | 2,062.36   | -0.85%                      |
| Price + volatility LSTM  | 1,514.03  | 2,074.88   | -1.69%                      |

The final evaluation covered **451 daily observations**, from **27 February 2025 to 23 May 2026**. LSTM predictions were averaged across three fixed random seeds.

Persistence achieved the lowest mean absolute error. The LSTM models did not consistently improve on this baseline.

The sentiment model was evaluated during development but was not included in the final evaluation. Historical sentiment features used the Fear and Greed index; collected news sentiment was processed separately.

These results do not establish statistical significance or trading profitability. The final period has now been inspected, so further model improvements require fresh evaluation data.

## Research report

Read the [research report](bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/docs/final-report.md) for methodology, results, and limitations.

## Dashboard screenshots

### Bitcoin price overview

![Bitcoin price overview](bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/docs/images/dashboard%20overview.png)

### Final evaluation

![Final evaluation](bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/docs/images/final%20evaluation.png)

### Earlier-period comparisons

![Earlier-period comparisons](bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/docs/images/Early%20period%20comparison.png)

### Forecast failure analysis

![Forecast failure analysis](bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/docs/images/Forecast%20failures.png)

### Synthetic public demo

This screenshot uses fictional data to demonstrate the interface.

![Synthetic public demo](bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/docs/images/Synthetic%20demo%20image.png)

## Run the dashboard

Open a terminal inside:

`bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting`

Create a virtual environment and install the dashboard dependencies:

```powershell
python -m venv C:\btc-env
C:\btc-env\Scripts\python.exe -m pip install -r requirements.txt
C:\btc-env\Scripts\python.exe -m streamlit run scripts\bitcoin_page.py
```

Use another short writable environment path if needed.

The public version starts with synthetic data. Historical research results require the original local archives and saved experiment outputs, which are excluded from this repository.

TensorFlow is optional for the dashboard demo. Training requires the additional dependencies in `requirements-research.txt`.

## Project structure

Project files are inside `bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/`.

- `scripts/` — dashboard, data processing, experiments, and checks.
- `src/cryptopulse/` — reused sentiment processing modules.
- `docs/` — research report, setup instructions, and screenshots.
- `work/` — ignored local research data and model outputs.

## Limitations

- Historical results do not demonstrate current forecasting ability.
- Performance varied across evaluation periods and random seeds.
- The models often predicted smaller movements than the market experienced.
- Historical sentiment availability and revisions cannot be fully verified.
- News sentiment scores describe language and do not establish future returns.
- The synthetic demo does not represent real market performance.
- Exact reproduction requires provider archives and saved research manifests.
- No trading strategy with fees and execution costs was evaluated.

## References

- [Reference Bitcoin prediction notebook](https://github.com/silvainfm/FinTech-Projects/blob/main/Stock%20Predictor/lstm_stock_predictor_fng.ipynb)
- [CryptoPulse AI](https://github.com/KaviyaVelmurugan/cryptopulse-ai) — sentiment processing reference.
- [SignalGuard](https://github.com/KaviyaVelmurugan/SignalGuard) — evaluation and failure-analysis reference.

## License

Project software is released under the [MIT License](LICENSE).

Reused components retain their original license notices. Provider data has separate terms; the MIT License does not grant permission to redistribute external datasets.

See [data and attribution](bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/DATA_AND_ATTRIBUTION.md) and [setup instructions](bitcoin-sentiment-forecasting-public/bitcoin-sentiment-forecasting/docs/SETUP.md).
