PYTHON ?= python

cython:
	find lda -name "*.pyx" -exec $(PYTHON)3 -m cython -3 {} \;
