# Capturing

**Capturing** is a small Python toolkit for **comparative text analysis**: load human-written and LLM-generated documents on a topic, measure similarity (TF–IDF + cosine similarity), estimate positions relative to human “baseline” perspectives (e.g. left / center / right), and visualize word distributions and document features.

## Layout

- `capturing/capturing.py` — Core `CapTuring` class (loading text, TF–IDF, similarity, Sankey and plots).
- `capturing/app.py` — CLI-style script that scans a topic folder and runs the full pipeline.
- `capturing/documents/<topic>/` — Per-topic corpora: `human/` (by perspective), `llm/` (by model), optional `support/stop_words.txt`.

Sample topics included: **ccp**, **roe_vs_wade**, **tariffs**.

## Requirements

- Python 3.10+ recommended

Install dependencies:

```bash
cd capturing
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
```

## Run

From the `capturing` directory (paths are relative to the current working directory):

```bash
cd capturing
python app.py
```

Edit the `TOPIC` variable at the top of `app.py` to switch between topics (e.g. `"ccp"`, `"roe_vs_wade"`, `"tariffs"`).

## License

This repository is provided as-is for research and educational use. Add a license file if you need explicit terms.
