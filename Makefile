PY ?= .venv/bin/python
LOGS = data/logs
REF = nav-id-hlr2e-3-detc-s*-7.5k-cpu-outer=nav-id:inf

.PHONY: setup test figures tables clean

setup:
	python3.13 -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q
	bash scripts/verify_repo.sh

figures:
	$(PY) scripts/plot_gen_curve.py --logs $(LOGS) --out docs/figures/partA_curve_k.png
	$(PY) scripts/plot_noise.py --out docs/figures/partB_noise.png

# tables 4-7 and 11-12 of the report
tables:
	mkdir -p results
	$(PY) scripts/summarize_gen_pilot.py 'gen-*' 'ent-*' --logs $(LOGS) --ref '$(REF)' > results/tables_4_7.txt
	$(PY) scripts/summarize_noise.py > results/tables_11_12.txt

clean:
	rm -rf results .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
