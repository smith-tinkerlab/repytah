Install from `pip` or `conda`
============

PyPI
~~~~

The latest stable release is available on PyPI, and you can install it by running::

    pip install repytah

Anaconda
~~~~~~~~

If you use Anaconda, you can install the package using ``conda-forge``::

    conda install -c conda-forge repytah

Source
~~~~~~

To build repytah from source, you need to first clone the repo using ``git clone git@github.com:smith-tinkerlab/repytah.git``. Then build with ``python setup.py build``. Then, to install repytah, say ``python setup.py install``.

Alternatively, you can download or clone the repository and use ``pip`` to handle dependencies::

    unzip repytah.zip
    pip install -e repytah

or::

    git clone https://github.com/smith-tinkerlab/repytah.git
    pip install -e repytah-main

By calling ``pip list`` you should see ``repytah`` now as an installed package::

    repytah (0.x.x, /path/to/repytah)

