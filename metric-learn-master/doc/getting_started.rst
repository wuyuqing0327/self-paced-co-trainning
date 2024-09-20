###############
Getting started
###############

Installation and Setup
======================

**Installation**

metric-learn can be installed in either of the following ways:

- If you use Anaconda: ``conda install -c conda-forge metric-learn``. See more options `here <https://github.com/conda-forge/metric-learn-feedstock#installing-metric-learn>`_.

- To install from PyPI: ``pip install metric-learn``.

- For a manual install of the latest code, download the source repository and run ``python setup.py install``. You may then run ``pytest test`` to run all tests (you will need to have the ``pytest`` package installed).

**Dependencies**

- Python 3.6+ (the last version supporting Python 2 and Python 3.5 was
  `v0.5.0 <https://pypi.org/project/metric-learn/0.5.0/>`_)
- numpy>= 1.11.0, scipy>= 0.17.0, scikit-learn>=0.21.3

**Optional dependencies**

- For SDML, using skggm will allow the algorithm to solve problematic cases
  (install from commit `a0ed406 <https://github.com/skggm/skggm/commit/a0ed406586c4364ea3297a658f415e13b5cbdaf8>`_).
  ``pip install 'git+https://github.com/skggm/skggm.git@a0ed406586c4364ea3297a658f415e13b5cbdaf8'`` to install the required version of skggm from GitHub.
- For running the examples only: matplotlib

Quick start
===========

This example loads the iris dataset, and evaluates a k-nearest neighbors
algorithm on an embedding space learned with `NCA`.

::

    from metric_learn import NCA
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsClassifier
    
    X, y = load_iris(return_X_y=True)
    clf = make_pipeline(NCA(), KNeighborsClassifier())
    cross_val_score(clf, X, y)
