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

We introduce `repytah`, a Python package that constructs aligned hierarchies, a structure-based representation for sequential data where repetitions have particular meaning (such as musical recordings or scores). Introduced by Kinnaird [@Kinnaird_ISMIR] and with music-based data as the primary motivation, the aligned hierarchies representation shows all possible hierarchical decompositions of a piece of music along a common time axis. The `repytah` package provides tools to extract repeated structures in sequential data (including music-based data), and offers a comprehensive mechanism to convert them into aligned hierarchies. This package is a translation of the original MATLAB code by Kinnaird [@Kinnaird_code]. 


# Statement of Need

Broadly, Music Information Retrieval (MIR) seeks to capture information about music. Content-based methods work directly on the recordings or scores, while other approaches leverage other kinds of information like metadata, tags, or listener surveys. There are a variety of tasks in MIR, including similarity tasks (like cover song identification and remix detection) that seek to determine how similar any two pieces of music are, and structure tasks (like the chorus detection task) that seek to label various structural features or artifacts. 

Music-based data streams often have repeated elements that build on each other, creating structure-based hierarchies. The aligned hierarchies representation by Kinnaird [@Kinnaird:2014] is a structure-based representation that combines the motivation of structure tasks with the goal of similarity tasks. A drawback of Kinnaird’s approach is that the original code was written in MATLAB. 

The Python package `repytah` forms the aligned hierarchies for a given sequential data stream. Similar to the original code, `repytah` extracts structural repetitions within a data stream and leverages their relationships to each other to build the aligned hierarchies representation. This encodes multi-scale pattern information and overlays all hierarchical decompositions of those patterns onto one object by aligning these hierarchical decompositions along a common time axis. 

`repytah` was completed, improved on, and successfully debugged by cross-referencing the desired output of the package with the output of the original MATLAB code. However, this package aims to give MIR users access to these tools through the open-source Python language instead of the proprietary MATLAB language. Additionally, this package provides more complete documentation, examples, and test files than the original code. 

There has been a long tradition of building Python packages for MIR research. Examples include the `AMEN` package [@amen], the `mir_eval` library [@Raffel14mir_eval:a], the `mirdata` library [@Bittner19mirdata], and the more recent `libfmp` package [@Müller2021]. As MIR has grown as a discipline, there has been a focus on reproducibility, accessibility, and open-source development. As such, there have been several examples of code being translated from MATLAB to python. The most notable example is `librosa`, a package that provides a number of powerful tools for MIR work [@McFee_librosa_SciPy]. 

# Functionality
There are four modules in the `repytah` Python package: transform, search, assemble, and utilities. Each module has an associated Jupyter notebook file that summarizes the module’s functions. There are also test files for each module. 

The four modules work together to form the aligned hierarchies, but each serves a slightly different purpose. Functions in the <ins>transform</ins> module transform matrix inputs into different forms, either from lists of indices into matrices and vice versa. The <ins>search</ins> module finds and records information about repeated structures, represented as diagonals in a song’s thresholded self-dissimilarity matrix. Once found, these repeated structures are later transformed and assembled into the aligned hierarchies using the <ins>assemble</ins> module, which finds the essential structure components from the repeated structures found with the search module, and then uses those essential structure components to build the aligned hierarchies. Lastly, the <ins>utilities</ins> module contains functions that are frequently called by functions in the other three modules. 

Additionally, the package includes <ins>example.py</ins>, which runs a complete example building aligned hierarchies when a CSV file with extracted features is the input.


# Acknowledgements
`repytah` was developed as part of Smith College's Summer Undergraduate Research Fellowship (SURF) in 2019, 2020 and 2021, and has been partially funded by Smith College's CFCD funding mechanism. Additionally, as Kinnaird is the Clare Boothe Luce Assistant Professor of Computer Science and Statistical & Data Sciences at Smith College, this work has also been partially supported by Henry Luce Foundation's Clare Boothe Luce Program.

# References

