VENV=venv/bin
PIP=$(VENV)/pip
PYTHON=$(VENV)/python
TRAIN=../deeploc_multi/results/cdhit_train_protvec.npz
TEST=../deeploc_multi/results/cdhit_test_protvec.npz
# TRAIN=../deeploc_multi/results/deeploc_train_onehot.npz
# TEST=../deeploc_multi/results/deeploc_test_onehot.npz
# TRAIN=./data/train.npz
# TEST=./data/test.npz

help:

clean: clean-venv
	rm -fv *.pyc

clean-venv:
	rm -rf venv/

# run the full model on cpu
run: | python
	$(PYTHON) train.py -i $(TRAIN) -t $(TEST)


# faster, but less accurate; only few epochs.  
rundev: | python
	$(PYTHON) train.py -i $(TRAIN) -t $(TEST) -e 3 -lr 0.01 -bs 64 -hn 128 -se 42

rundevgpu: | python
	THEANO_FLAGS=device=gpu0,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once,warn_float64=warn $(PYTHON) train.py -i $(TRAIN) -t $(TEST) -e 1

# run the full model on gpu
rungpu: | python
	THEANO_FLAGS=device=gpu0,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once,warn_float64=warn $(PYTHON) train.py -i $(TRAIN) -t $(TEST)

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


