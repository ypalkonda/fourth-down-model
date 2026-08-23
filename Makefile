# Entry points for the fourth-down model.
#
# PYTHONPATH=src rather than relying on `pip install -e .`: on macOS, pip sets
# the UF_HIDDEN file flag on the .pth files it writes into site-packages, and
# Python 3.14's site.addpackage() skips hidden .pth files. The editable install
# therefore resolves at install time and then silently stops working. Setting
# PYTHONPATH explicitly sidesteps the whole interaction.

PY := ./.venv/bin/python
export PYTHONPATH := src

.PHONY: setup data train-state train-outcomes train-baseline score sanity report api dashboard dashboard-build dashboard-shots all clean

setup:
	./setup.sh

data:
	$(PY) scripts/build_data.py

train-state:
	$(PY) scripts/train_state.py

train-baseline:
	$(PY) scripts/train_baseline.py

train-outcomes:
	$(PY) scripts/train_outcomes.py

score:
	$(PY) scripts/score_history.py

sanity:
	$(PY) scripts/sanity_check.py

report:
	$(PY) scripts/validation_report.py

api:
	$(PY) -m api.app

# The dashboard uses a locally installed Node so nothing has to be on PATH.
NODE := $(HOME)/.local/node/bin

dashboard:
	cd dashboard && PATH="$(NODE):$$PATH" npm run dev

dashboard-build:
	cd dashboard && PATH="$(NODE):$$PATH" npm install --no-fund --no-audit && PATH="$(NODE):$$PATH" npm run build

dashboard-shots:
	cd dashboard && PATH="$(NODE):$$PATH" node shoot.mjs

# Full rebuild from raw data to a served dashboard.
all: data train-state train-outcomes score dashboard-build

clean:
	rm -rf data/processed/*.parquet models/*.pt models/*.json models/*.npz
