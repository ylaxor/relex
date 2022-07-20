![alt text](relex.png)

[![PyPI version](https://badge.fury.io/py/relex.svg)](https://badge.fury.io/py/relex)
[![GitHub Issues](https://img.shields.io/github/issues/ylaxor/relex.svg)](https://github.com/ylaxor/relex/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![docs](https://readthedocs.org/projects/docs/badge/?version=latest)](https://relex-docs.readthedocs.io/en/latest/index.html)


Relex is an open-source project & python package, aimed to provide easy-to-use pipelines for building custom and deep-learning based semantic relation extraction systems.

---

Relationship extraction is the task of extracting semantic relationships from a text. Extracted relationships usually occur between two or more entities of a certain type (e.g. Person, Country, Instrument, Region) and fall into a number of semantic categories (e.g. has nationality, plays, located in).


Now at [version 0.1.1](https://github.com/ylaxor/relex/releases)!

## Setup

Requires python 3.7.13 or greater.

```
pip install relex
```

## Motivations and philosphy of the project

* More details on the [relex official documentation](https://relex-docs.readthedocs.io/en/latest/philosophy.html).

## Package documentation and supported use cases

* [Technical details](https://relex-docs.readthedocs.io/en/latest/main_sections.html#package-technical-details).
* [Example of effective use cases](https://relex-docs.readthedocs.io/en/latest/usage.html).

## Contributing

Before contributing, please read carefully the Context and motivation section, more precisely the paragraph: Bringing together applied research and good coding practice for more details about relex (the project) philosophy and its (the package) design. Therein we attempted to motivate the adopted structure of the package, the role of each sub-package, and we tried to share our vision of how the project can evolve in the future and become more useful.

**Contributing to improve code quality:**

- Report Bugs in existing features
- Fix reported Bugs
- Improve existing features
- Suggest new features
- Implement new features
- Write tests
- Write documentation

**Contributing as applied (NLP) researcher:**

- All the previous
- Suggest new pipelines
- Implement new pipelines

Relex lives, by design, on applied research in (NLP).

We think that any standard deep relation extraction problem can be modelled and successfully solved under the relex package’s high-level design, which adheres to the modular workflow described in the Context and motivation section of the documentation.

People are encouraged to participate and make their ideas a tangible part of relex if they conduct research around deep relation extraction and have fresh pipelines in mind that adhere to relex’s spirit (project, and package). In that case, a pipeline would be a collection of routines (data loaders, architectures, learners, predictors, and other miscellaneous utilities) that should each be considered to fit the general deep-learning based framework adopted in relex (c.f. Simplifying the relation extraction problem and following paragraphs) and as a result be able to find its place in the corresponding relex sub-package.

*Contributions are welcomed and much valued! Credit is always given, and every little bit helps.*

[Updated list of contributors](https://relex-docs.readthedocs.io/en/latest/credits.html#list-of-contributors)


## Contact

Please email your feedbacks to [Ali NCIBI](mailto:contact@alincibi.me).

## License

The MIT License (MIT)

Relex is licensed under the following MIT license: The MIT License (MIT) Copyright © 2022 Ali NCIBI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.