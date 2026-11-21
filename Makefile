PYTHON ?= python3.11
ML_DATA ?= data/screening/dev

.PHONY: data match eval alerts app test ml-check ml-train ml-predict ml-validate ml-finetune ml-all
ml-check:
	$(PYTHON) -m screening_ml.check_data --data $(ML_DATA)
ml-train:
	$(PYTHON) -m screening_ml.train
ml-predict:
	$(PYTHON) -m screening_ml.predict --data $(ML_DATA) --out artifacts/dev_nn.jsonl
ml-validate:
	$(PYTHON) data/screening/validate.py --data $(ML_DATA) --output artifacts/dev_nn.jsonl --score $(ML_DATA)/practice_labels.jsonl
# submission: rule engine, standard library only
submit-run:
	$(PYTHON) system/run $(ML_DATA) output/dev.jsonl
demo:
	$(PYTHON) -m screening_ml.demo_server --data $(ML_DATA) --port 8765
analyst-view:
	$(PYTHON) -m screening_ml.analyst_view --data $(ML_DATA) --out artifacts/analyst_view.html
submit-validate:
	$(PYTHON) data/screening/validate.py --data $(ML_DATA) --output output/dev.jsonl --score $(ML_DATA)/practice_labels.jsonl
ml-finetune:
	$(PYTHON) -m screening_ml.finetune
ml-all: ml-check ml-train ml-predict ml-validate

data:
	$(PYTHON) -m src.load_list
	$(PYTHON) -m src.synth
match:
	$(PYTHON) -m src.baseline
	$(PYTHON) -m src.match
eval:
	$(PYTHON) -m src.evaluate
alerts:
	$(PYTHON) -m src.build_alerts $(ALERT_ARGS)
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
test:
	$(PYTHON) -m pytest -q
