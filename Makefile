PYTHON ?= python3.11

.PHONY: data match eval alerts app test
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
