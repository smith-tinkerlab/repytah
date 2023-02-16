# <img alt="repytah" src="branding/repytah_logo.png" height="100">

A Python package that builds aligned hierarchies for sequential data streams.

[![PyPI](https://img.shields.io/pypi/v/repytah.svg)](https://pypi.python.org/pypi/repytah)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/repytah/badges/version.svg)](https://anaconda.org/conda-forge/repytah)

[![License](https://img.shields.io/pypi/l/repytah.svg)](https://github.com/smith-tinkerlab/repytah/blob/main/LICENSE.md)
[![CI](https://github.com/smith-tinkerlab/repytah/actions/workflows/check_repytah.yml/badge.svg)](https://github.com/smith-tinkerlab/repytah/actions/workflows/check_repytah.yml)

[![codecov](https://codecov.io/gh/tinkerlab/repytah/branch/main/graph/badge.svg?token=ULWnUHaIJC)](https://codecov.io/gh/tinkerlab/repytah)

## Documentation

See our [website](https://repytah.readthedocs.io/en/latest/index.html) for a complete reference manual and introductory tutorials.

This [example](https://repytah.readthedocs.io/en/latest/example_vignette.html) tutorial will show you a usage of the package from start to finish.

## Statement of Need

### Problems Addressed

Sequential data streams often have repeated elements that build on each other, creating hierarchies. Therefore, the goal of `repytah` is to extract these repetitions and their relationships to each other in order to form aligned hierarchies.

To learn more about aligned hierarchies, see this [paper](https://s18798.pcdn.co/ismir2016/wp-content/uploads/sites/2294/2016/07/020_Paper.pdf) by Kinnaird (ISMIR 2016) which introduces aligned hierarchies in the context of music-based data streams.

### Audience

People working with sequential data where repetitions have meaning will find `repytah` useful including computational scientists, advanced undergraduate students, younger industry experts, and many others.

An example application of `repytah` is in Music Information Retrieval (MIR), i.e., in the intersection of music and computer science.

## Installation

The latest stable release is available on PyPI, and you can install it by running:

```bash
pip install repytah
```

If you use Anaconda, you can install the package using `conda-forge`:

```bash
conda install -c conda-forge repytah
```

To build repytah from source, say `python setup.py build`.
Then, to install repytah, say `python setup.py install`.

Alternatively, you can download or clone the repository and use `pip` to handle dependencies:

```bash
unzip repytah.zip
pip install -e repytah
```

or

```bash
git clone https://github.com/smith-tinkerlab/repytah.git
pip install -e repytah
```

By calling `pip list` you should see `repytah` now as an installed package:

```bash
repytah (0.x.x, /path/to/repytah)
```

## Current and Future Work - Elements of the Package

* Aligned Hierarchies - This is the fundamental output of the package, of which derivatives can be built. The aligned hierarchies for a given sequential data stream is the collection of all possible **hierarchical** structure decompositions, **aligned** on a common time axis. To this end, we offer all possible structure decompositions in one cohesive object.
  * Includes walk through file example.py using supplied input.csv
* _Forthcoming_ Aligned sub-Hierarchies - (AsH) - These are derivatives of the aligned hierarchies and are described in [Aligned sub-Hierarchies: a structure-based approach to the cover song task](http://ismir2018.ircam.fr/doc/pdfs/81_Paper.pdf)
* _Forthcoming_ Start-End and S_NL diagrams
* _Forthcoming_ SuPP and MaPP representations

### MATLAB code

The original code to this project was written in MATLAB by Katherine M. Kinnaird. It can be found [here](https://github.com/kmkinnaird/ThesisCode).

### Acknowledgements

This code was developed as part of Smith College's Summer Undergraduate Research Fellowship (SURF) from 2019 to 2022, and has been partially funded by Smith College's CFCD funding mechanism. Additionally, as Kinnaird is the Clare Boothe Luce Assistant Professor of Computer Science and Statistical & Data Sciences at Smith College, this work has also been partially supported by Henry Luce Foundation's Clare Boothe Luce Program.

Additionally, we would like to acknowledge and give thanks to Brian McFee and the [librosa](https://github.com/librosa) team. We significantly referenced the Python package [librosa](https://github.com/librosa/librosa) in our development process.

### Citing

Please cite `repytah` using the following:

C. Jia et al., repytah: A Python package that builds aligned hierarchies for sequential data streams. Python package version 0.1.0, 2023. [Online]. Available: [https://github.com/smith-tinkerlab/repytah](https://github.com/smith-tinkerlab/repytah).
