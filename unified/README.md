# Unified pipeline
- `unified_analysis.ipynb` — executed notebook: every result and figure in the deck, in slide order.
- `metnet/` — `graph.py` (one graph builder, filtering = switches), `regression.py` (all estimators), `simulation.py`, `plots.py`/`theme.py` (deck theme, Poppins).
- `figures/`, `results/` — outputs. `build_notebook.py` regenerates the notebook.

Run: `uv venv .venv && uv pip install --python .venv/bin/python numpy pandas scipy matplotlib openpyxl jupyter nbconvert`, then
`python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace unified_analysis.ipynb` (~2 min).
Original BiGG JSON files (for subsystems) are downloaded into `../data/old_bigg/` if missing.
