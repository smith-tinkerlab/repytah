# repytah

A Python package for building Aligned Hierarchies for music-based data streams.

<!-- Badges
[![PyPI](https://img.shields.io/pypi/v/librosa.svg)](https://pypi.python.org/pypi/librosa)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/librosa/badges/version.svg)](https://anaconda.org/conda-forge/librosa)
[![License](https://img.shields.io/pypi/l/librosa.svg)](https://github.com/librosa/librosa/blob/main/LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)


## Elements of the package

[![CI](https://github.com/librosa/librosa/actions/workflows/ci.yml/badge.svg)](https://github.com/librosa/librosa/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/librosa/librosa/branch/main/graph/badge.svg?token=ULWnUHaIJC)](https://codecov.io/gh/librosa/librosa) -->

For details on the aligned hierarchies, see [Aligned Hierarchies: A Multi-scale structure-based representation for music-based data streams](https://s18798.pcdn.co/ismir2016/wp-content/uploads/sites/2294/2016/07/020_Paper.pdf) by Kinnaird (ISMIR 2016).

## Documentation

See (link to website) for a complete reference manual and introductory tutorials.

This [example](link to example vignette) tutorial will show you a usage of the package from start to finish.

## Statement of Need

### Problems addressed

Music-based data streams often have repeated elements that build on each other, creating hierarchies. Therefore, the goal of the Python package repytah is to extract these repetitions and their relationships to each other in order to form aligned hierarchies.

### Audience
The target audience is people who are working with sequential data where repetitions have meaning: computational researchers, advanced undergraduate students who can be seen as younger industry esperts, etc. For example, people who are both music lovers and know computer science are our audiences since they will be interested in knowing the implication and application of Python in music.


## Installation

The latest stable release is available on PyPI, and you can install it by running:

```bash
pip install repytah
```

Anaconda users can install using `conda-forge`:

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


* Aligned Hierarchies - This is the fundamental output of the package, of which derivatives can be built. The aligned hierarchies for a given music-based data stream is the collection of all possible **hierarchical** structure decompositions, **aligned** on a common time axis. To this end, we offer all possible structure decompositions in one cohesive object.
  * Includes walk through file example.py using supplied input.csv
  * _Forthcoming_ Distance metric between two aligned hierarchies
* _Forthcoming_ Aligned sub-Hierarchies - (AsH) - These are derivatives of the aligned hierarchies and are described in [Aligned sub-Hierarchies: a structure-based approach to the cover song task](http://ismir2018.ircam.fr/doc/pdfs/81_Paper.pdf)
  * _Forthcoming_ Distance metric between two AsH representations
* _Forthcoming_ Start-End and S_NL diagrams
* _Forthcoming_ SuPP and MaPP representations

<!-- this block should be part of documentation website

### Modules

* [Aligned Hierarchies](https://github.com/smith-tinkerlab/ah/tree/master/aligned-hierarchies)
  * [example.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/example.py) - Includes a complete aligned hierarchies case in which a csv file with extracted features is input and the aligned  hierarchies are output.
  * [utilities.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/utilities.py) - Includes utility functions that allow larger functions in other modules to function.
  * [search.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/search.py) - Includes functions that find structures and information about those structures.
  * [transform.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/transform.py) - Includes functions that transform inputs to be of use in larger functions in assemble.py.
  * [assemble.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/assemble.py) - Includes functions that create the hierarchical structure and build the aligned hierarchies. -->

### MATLAB code

The original code to this project was written in MATLAB by Katherine M. Kinnaird. It can be found [here](https://github.com/kmkinnaird/ThesisCode).

### Acknowledgements

This code was developed as part of Smith College's Summer Undergraduate Research Fellowship (SURF) from 2019 to 2021, and has been partially funded by Smith College's CFCD funding mechanism. Additionally, as Kinnaird is the Clare Boothe Luce Assistant Professor of Computer Science and Statistical & Data Sciences at Smith College, this work has also been partially supported by Henry Luce Foundation's Clare Boothe Luce Program.

### Citing

Please cite `repytah` using the following:

K. M. Kinnaird, et al. repytah: Python package for building Aligned Hierarchies for music-based data streams. Python package version 0.0.0-alpha, 2021. [Online]. Available: [https://github.com/smith-tinkerlab/ah](https://github.com/smith-tinkerlab/ah).
