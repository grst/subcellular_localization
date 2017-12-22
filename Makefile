VENV=venv/bin
PIP=$(VENV)/pip
PYTHON=$(VENV)/python

help:

clean: clean-venv

clean-venv:
	rm -rf venv/

run: | python
	$(PYTHON) train.py -i ./data/train.npz -t ./data/test.npz

################################################################################
# Setup python virtual environment
################################################################################

.PHONY: python
python: $(PYTHON)

venv:
	virtualenv venv -p /usr/bin/python2

$(PIP): | venv
	$(PIP) install --upgrade pip

venv/.installed: requirements-dev.txt | venv
	$(PIP) install -Ur requirements-dev.txt
	# $(PIP) install -e .

$(PYTHON): | $(PIP) venv/.installed


