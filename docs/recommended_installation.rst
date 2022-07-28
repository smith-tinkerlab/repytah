Recommended Installation
============

Anaconda
~~~~~~~~

The safest way to install repytah is through ``conda``.
In terminal, navigate to the directory that contains environment.yml and execute::

    conda env create -f environment.yml
    conda activate repytahenv
    conda install -c conda-forge repytah

PyPI
~~~~

In terminal, navigate to the directory that contains `requirements.txt` and execute the following command::

    pip install -r requirements.txt
    pip install repytah

Warning: Might encounter problems if Python version is not >= 3.7, <3.11.
