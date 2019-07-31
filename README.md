# ah package

This python package builds the Aligned Hierarchies for music-based data streams. For details on the aligned hierarchies, see [Aligned Hierarchies: A Multi-scale structure-based representation for music-based data streams](https://s18798.pcdn.co/ismir2016/wp-content/uploads/sites/2294/2016/07/020_Paper.pdf) by Kinnaird (ISMIR 2016).


## Elements of the package

* Aligned Hierarchies - This is the fundamental output of the package, of which derivatives can be built. The aligned hierarchies for a given music-based data stream is the collection of all possible **hierarchical** structure decompositions, **aligned** on a common time axis. To this end, we offer all possible structure decompositions in one cohesive object.
    * Includes walk through file example.py using supplied input.csv
    * _Forthcoming_ Distance metric between two aligned hierarchies
* _Forthcoming_ Aligned sub-Hierarchies - (AsH) - These are derivatives of the aligned hierarchies and are described in [Aligned sub-Hierarchies: a structure-based approach to the cover song task](http://ismir2018.ircam.fr/doc/pdfs/81_Paper.pdf)
    * _Forthcoming_ Distance metric between two AsH representations
* _Forthcoming_ Start-End and S_NL diagrams
* _Forthcoming_ SuPP and MaPP representations

### Modules

* [Aligned Hierarchies](https://github.com/smith-tinkerlab/ah/tree/master/aligned-hierarchies)
   * [example.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/example.py) - Includes a complete aligned hierarchies case in which a csv file with extracted features is input and the aligned  hierarchies are output.
   * [utilities.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/utilities.py) - Includes utility functions that allow larger functions in other modules to function.
   * [search.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/search.py) - Includes functions that find structures and information about those structures.
   * [transform.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/transform.py) - Includes functions that transform inputs to be of use in larger functions in assemble.py.
   * [assemble.py](https://github.com/smith-tinkerlab/ah/blob/master/aligned-hierarchies/assemble.py) - Includes functions that create the hierarchical structure and build the aligned hierarchies.

## Contributors

The contributors for this package are:
* Aligned Hierarchies
    * [Jordan Moody](https://github.com/jormacmoo)
    * [Lizette Carpenter](https://github.com/lcarpenter20)
    * [Eleanor Donaher](https://github.com/edonaher)
    * [Denise Nava](https://github.com/d-nava)


### Matlab code

Original MATLAB code by Kinnaird can be found [here](https://github.com/kmkinnaird/ThesisCode).

### Funding sources

This code was developed as part of Smith College's Summer Undergraduate Research Fellowship (SURF) in 2019 and was partially funded by the college's CFCD funding mechanism.
