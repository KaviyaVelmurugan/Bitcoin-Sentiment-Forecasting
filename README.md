# Bitcoin Sentiment Forecasting

A research dashboard comparing next-day Bitcoin forecasts with persistence: tomorrow equals today. A fictional public demo runs without credentials, private files or training.

## Quick start

Use Python 3.13 on Windows x64. From this project folder, create an environment at a short writable path to avoid Windows long-path installation failures:

```powershell
python -m venv C:\btc-env
C:\btc-env\Scripts\python.exe -m pip install -r requirements.txt
C:\btc-env\Scripts\python.exe -m streamlit run scripts\bitcoin_page.py
```

Choose another short writable path if needed. On macOS/Linux use a standard virtual environment and equivalent Python commands. TensorFlow is optional for the demo; install `requirements-research.txt` only for training.

On a clean checkout, the page shows **Synthetic demo**. Prices, forecasts and news counts there are entirely fictional. They demonstrate the interface, not market performance. Existing local users can choose **Local research results** when their saved archive is present.

## Historical result

| Forecast | Final MAE (USD) | Improvement vs baseline |
|---|---:|---:|
| Persistence | 1,488.92 | 0.00% |
| Price-only LSTM | 1,501.59 | -0.85% |
| Price + volatility LSTM | 1,514.03 | -1.69% |

The final period contained 451 observations, 27 February 2025 to 23 May 2026. LSTM forecasts average three fixed seeds. Persistence achieved the lowest final MAE. No statistical significance, live forecasting or trading-profit claim is made. The final period is now inspected: future changes require new evaluation data.

[Research report](docs/final-report.md) explains methodology and limits. Historical findings are separate from synthetic demo charts. Exact reproduction requires original provider archives and saved manifests excluded from the public demo.

## Layout

- `scripts/`: dashboard, collection, processing, experiments and checks.
- `src/cryptopulse/`: reused sentiment modules with their original MIT notice.
- `docs/`: setup, final report and publication notes.
- `work/`: private local archives, news, models and results, ignored by Git.

## References and license

[Reference notebook](https://github.com/silvainfm/FinTech-Projects/blob/main/Stock%20Predictor/lstm_stock_predictor_fng.ipynb) inspired the experiment. [CryptoPulse AI](https://github.com/KaviyaVelmurugan/cryptopulse-ai) supplies sentiment code; [SignalGuard](https://github.com/KaviyaVelmurugan/SignalGuard) informs evaluation and failure analysis. News was not used in the historical forecasting models.

Software is [MIT licensed](LICENSE). Provider datasets retain separate licenses; MIT does not grant rights to redistribute them. See [attribution](DATA_AND_ATTRIBUTION.md) and [setup](docs/SETUP.md). Manual news collection remains subject to provider and publisher permissions; do not upload keys or private news.
