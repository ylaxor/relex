Before contributing, please refer to the `context and motivation <./philosophy.html>`_ section, more precisely to the paragraph: `Bringing together applied research and good coding practice <./philosophy.html#bringing-together-applied-research-and-good-coding-practice>`_ for more details about relex (the project) philosophy and its (the package) design. Therein we attempted to motivate the adopted structure of the package, the role of each sub-package, and we tried to share our vision of how the project can evolve in the future and become more useful.

Contributing to improve code quality
======================================


* Report Bugs in existing features
* Fix reported Bugs
* Improve existing features
* Suggest new features
* Implement new features
* Write tests
* Write documentation

Contributing as applied (NLP) researcher
=========================================

All the previous, but essentially:

* Suggest new pipelines
* Implement new pipelines

Relex lives, by design, on applied research in (NLP).

We think that any standard deep relation extraction problem can be modelled and successfully solved under the relex package's high-level design, which adheres to the modular workflow described in the `Context and motivation <./philosophy.html>`_  section of the documentation.

People are encouraged to participate and make their ideas a tangible part of relex if they conduct research around deep relation extraction and have fresh pipelines in mind that adhere to relex's spirit (project, and package).
In that case, a pipeline would be a collection of routines (data loaders, architectures, learners, predictors, and other miscellaneous utilities) that should each be considered to fit the general deep-learning based framework adopted in relex (c.f. `Simplifying the relation extraction problem <./philosophy.html#simplifying-the-relation-extraction-problem>`_ and following paragraphs) and as a result be able to find its place in the corresponding relex sub-package.

Contributions are welcomed and much valued! Credit is always given, and every little bit helps.

* `Updated list of contributors <./credits.html#list-of-contributors>`_

