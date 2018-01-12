VENV=venv/bin
PIP=$(VENV)/pip
PYTHON=$(VENV)/python

help:

clean: clean-venv

clean-venv:
	rm -rf venv/

run: | python
	$(PYTHON) train.py -i ./data/train.npz -t ./data/test.npz

rungpu: | python
	THEANO_FLAGS=device=gpu0,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once,warn_float64=warn $(PYTHON) train.py -i ./data/train.npz -t ./data/test.npz

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


