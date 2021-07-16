---
title: '`repytah`: An Open-Source Python Package That Builds Aligned Hierarchies for Sequential Data Streams'
tags:
  - Python
  - Music Information Retrieval
  - Structure representations
  - Aligned Hierarchies

authors:
  - name: Adrian M. Price-Whelan^[Custom footnotes for e.g. denoting who the corresspoinding author is can be included like this.]
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: Name, Position (if relevant), Affiliation
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: DD Month YYYY
bibliography: joss.bib

---

# Summary

Broadly, Music Information Retrieval (MIR) seeks to capture information about music. Content-based methods work directly on the recordings or scores, while other approaches leverage other kinds of data such as metadata, tags, or surveys. Some MIR tasks seek to determine how similar any two pieces of music are, while others seek to label various structural features. The Aligned Hierarchies representation by Kinnaird [@Kinnaird: 2014] is a structure-based representation that can be used to show how similar two pieces of music are. This representation combines the motivation of structure tasks with the goal of similarity tasks. A drawback of Kinnaird’s approach is that the original code was written in MATLAB. 
We introduce `repytah`, a Python package that constructs aligned hierarchies. The `repytah` package provides tools to extract repeated structures in musical data, and offers a comprehensive mechanism to convert music-based data streams to hierarchical structures. The original code was written in MATLAB based upon the PhD research of Katherine M. Kinnaird[@Kinnaird: 2014]. 


# Statement of Need

There has been a long tradition of building Python packages for MIR. Particularly, there are packages that involve converting MATLAB code to Python. `librosa`, a package that provides powerful tools to establish MIR systems, is one of them [@brian_mcfee-proc-scipy-2015]. Other examples include the FMP Notebooks [@MuellerZ19_FMP_ISMIR], the AMEN package [@amen], the mir_eval library [@Raffel14mir_eval:a] , and the mirdata library [@Bittner19mirdata].

Music-based data streams often have repeated elements that build on each other, creating hierarchies. Therefore, the goal of the Python package `repytah` is to extract these repetitions and their relationships to each other in order to form aligned hierarchies. This encodes multi-scale pattern information and overlays all hierarchical decompositions of those patterns onto one object by aligning these hierarchical decompositions along a common time axis. This package is written based on Katherine M. Kinnaird’s thesis *Aligned Hierarchies for Sequential Data* and the accompanying MATLAB code[@Kinnaird: 2014]. 

This package was written to give MIR users access to these tools through the open-source Python language instead of the proprietary MATLAB language. The Python package was completed, improved on, and successfully debugged by cross-referencing the desired output of the package with the output of the MATLAB code.

# Functionality
There are four modules in the `repytah` Python package:

- utilities.py: This module includes functions that are frequently called by larger functions in other modules so that the entire package can run smoothly.
- search.py: This module includes functions that find and record information about repeat structures in the form of diagonals in the threshold matrix. Once found, these repeat structures can later be transformed and assembled.
- transform.py: This module includes functions that transform matrix inputs into different forms that are of use when being called by larger functions.
- assemble.py: This module includes functions that find and form the essential structure components used to build the aligned hierarchies. 

Besides these four modules, the package also includes example.py, which is an example module that runs a complete case of building aligned hierarchies when a CSV file with extracted features is the input.

The Jupyter notebook files were also generated that summarize what these modules consist of, which together serve as a guide through the package. There is a distinct notebook file for each module, as well as an overarching file highlighting the code from start to finish. There are also test files for each module to ensure that the functions work as expected.

# Acknowledgements
`repytah` was developed as part of Smith College's Summer Undergraduate Research Fellowship (SURF) in 2019, 2020 and 2021, and has been partially funded by Smith College's CFCD funding mechanism. Additionally, as Kinnaird is the Clare Boothe Luce Assistant Professor of Computer Science and Statistical & Data Sciences at Smith College, this work has also been partially supported by Henry Luce Foundation's Clare Boothe Luce Program.

# References

