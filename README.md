# CRSQ: Chemical Reaction Simulator Q

This is a python program that constructs a chemical simulator quantum circuit on
the quantum computer SDK, Qiskit, from IBM.

Portions of the circuit can be run on the qiskit Aer simulator.

Jupyter notebooks are provided in the crsq-papers repository which show the circuit
diagrams and simulation results.

## Platform

The tested platform is Ubuntu22.04 running on WSL2 on Windows11.

## Setup

Here we describe how to set up a running environment and view the jupyter notebooks.

1. Install requisite software

python3 that comes with Ubuntu is required.
```bash
$ apt update
$ apt install python3
```

If you are using WSL2, the "wslu" package is recommended.  This will allow jupyter notebook to launch a browser on the Windows side from within WSL2.
```bash
$ apt install wslu
```

1. Make a directory to work in.  Here we will name it "crsq".

```bash
$ mkdir crsq
$ cd crsq
```

2. clone the repositories crsq-heap, crsq-arithmetic, crsq-main, crsq-papers in that directory.

```bash
$ git clone https://github.com/crsq-dev/crsq-heap.git
$ git clone https://github.com/crsq-dev/crsq-arithmetic.git
$ git clone https://github.com/crsq-dev/crsq-main.git
$ git clone https://github.com/crsq-dev/crsq-papers.git
```

3. make a python virtual environment for these projects at the crsq directory.

```bash
$ python3 -m venv .venv
```

4. Activate the venv and install the packages listed inside crsq-papers/2023

```bash
$ . .venv/bin/activate
$ pip install -r crsq-papers/2023/requirements.txt
```

5. set PYTHONPATH so that all the "src" directories will be listed.
```bash
$ export PYTHONPATH=$PWD/crsq-heap/src:$PWD/crsq-arithmetic/src:$PWD/crsq-main/src
```

Note: if you want to run the tests, add the test directories as well.

Jupyter notebook files are provided in the crsq-papers repository.
See that directory for details.
