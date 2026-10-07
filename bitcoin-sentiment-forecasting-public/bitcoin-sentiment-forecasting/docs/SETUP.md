# Setup

The public demo requires Python and `requirements.txt`. It reads no private files or API data. Follow the README, then launch `python -m streamlit run scripts/bitcoin_page.py` using your environment interpreter.

Open the workspace file in VS Code and choose **Python: Select Interpreter**. The workspace no longer contains a machine-specific path. The author's existing short-path environment still works; no reinstall or retraining is needed locally.

## Optional research

Install `requirements-research.txt` for TensorFlow training. Research scripts need original local archives, manifests and prior outputs in ignored `work/`. A clean checkout can run the public demo but cannot reproduce exact historical results without those inputs.

CryptoPulse modules are bundled under `src/cryptopulse`. Use `scripts/collect_news.py --prompt-key` for manual local-development collection, subject to provider/publisher rights. Keys are not saved by the prompt. Scoring news does not automatically make it eligible for model training.

Preserve the frozen final results separately; do not tune against that inspected period.
