Supported Python versions
==========================
Python ``3.7.13`` or a more recent version is required to setup and use relex.

Requirements
==============
Handling of deep neural networks and of their learning processes in relex is done basically using pytorch as backend environment, along with the following amazing packages:

- `flair <https://github.com/flairNLP/flair>`_, a Pytorch-based library for high-level (NLP).
- `torchmetrics <https://torchmetrics.readthedocs.io/>`_, a Python library for creating evaluation metrics.
- `numpy <https://www.numpy.org/>`_, a Python library for scientific computing.
- `pandas <https://pandas.pydata.org/>`_, a Python library for data manipulation and analysis.

Installation
==============
Relex package can be easily installed with: 

.. code-block:: console

    $ pip install relex

Sub-packages
=====================
To achieve its goal of being a simple and easy-to-use framework for building relation extraction systems, relex (the package) has been thought to include the following tree of sub-packages:

.. toctree::
   :maxdepth: 4

   relex.data
   relex.models
   relex.learners
   relex.predictors
   relex.utilities

More details about the motivations behind the choice of this package structure and the purpose of each sub-package can be found in this `paragraph <./philosophy.html#bringing-together-applied-research-and-good-coding-practice>`_.
Please refer to `contributions <./contributions.html>`_  guidelines for more information on how to contribute to relex.
