---
title: '`repytah`: An Open-Source Python Package for Building Aligned Hierarchies for Sequential Data'
tags:
  - Python
  - Music Information Retrieval
  - Structure representations
  - Aligned Hierarchies
  - Music Structure Analysis

authors:
  - name: Chenhui Jia
    affiliation: 1
  - name: Lizette Carpenter
    affiliation: 1
  - name: Thu Tran
    affiliation: 1
  - name: Amanda Y. Liu
    affiliation: 1
  - name: Sasha Yeutseyeva
    affiliation: 1
  - name: Marium Tapal
    orcid: 0000-0001-5093-6462
    affiliation: 1
  - name: Yingke Wang
    affiliation: 2
  - name: Zoie Kexin Zhao
    affiliation: 1
  - name: Jordan Moody
    affiliation: 1
  - name: Denise Nava
    affiliation: 1
  - name: Eleanor Donaher
    affiliation: 1
  - name: Lillian Yushu Jiang
    affiliation: 1
  - name: Ben Bruncati
    affiliation: 1
  - name: Katherine M. Kinnaird
    orcid: 0000-0002-0435-8996
    corresponding: true 
    affiliation: 1
affiliations:
 - name: Smith College, USA
   index: 1
 - name: Columbia University, USA
   index: 2
date: DD Month YYYY
bibliography: joss.bib

---

# Summary

We introduce `repytah`, a Python package that constructs the aligned hierarchies representation that contains all possible structure-based hierarchical decompositions for a piece of sequential data aligned on a common time axis. In particular, this representation, introduced by Kinnaird [@Kinnaird_ah] with music-based data (like musical recordings or scores) at the primary motivation, is intended for sequential data where repetitions have particular meaning (such as a verse, chorus, motif, or theme). The `repytah` package builds these aligned hierarchies by first extracting repeated structures (of all meaningful lengths) from sequential data. This package is a Python translation of the original MATLAB code by Kinnaird [@Kinnaird_code] with additional documentation, and the code has been updated to leverage efficiencies in Python.


# Statement of Need

Broadly, Music Information Retrieval (MIR) seeks to capture information about music in pursuit of various music-related tasks, such as playlist recommendation, cover song identification, and beat detection. Content-based methods work directly on musical artifacts such as audio recordings or musical scores, while other approaches leverage information about musical artifacts such as metadata (like the artist or album name), tags (like genre), or listener surveys. Approaches to MIR tasks can also include a focus on repeated (or novel) elements. 

Music-based data streams, like music, often have repeated elements that build on each other, creating structure-based hierarchies. The aligned hierarchies representation by Kinnaird [@Kinnaird_ah:2016] is a structure-based representation that places all possible structural-hierarchical decompositions of a data stream onto one common time axis. These aligned hierarchies when applied to music-based data streams can be used for visualization or for similarity tasks within MIR like the fingerprint task [@Kinnaird_ah]. The aligned hierarchies can also be post-processed to address additional tasks such as the cover song task [@Kinnaird_ash; @snl; @supp]. A drawback of Kinnaird’s code, however, is that the original code was written in MATLAB, which can only be used with a license and hence is not broadly accessible. 

There has been a long tradition of building Python packages for MIR research. Examples include the `AMEN` package [@amen], the `mir_eval` library [@mir_eval] , the `mirdata` library [@mirdata], and the more recent `libfmp` package [@libfmp]. As MIR has grown as a discipline, there has been an increased focus on reproducibility, accessibility, and open-source development [@reproducibility_mir, @mirdata]. As such, there have been several examples of code being translated from MATLAB to Python. The most notable example is `librosa`, a package that provides a number of powerful tools for MIR work [@librosa]. 

Building on this tradition, the presented package `repytah` forms the aligned hierarchies for a given sequential data stream, giving MIR users broader access to these tools through the open-source Python language. Similar to the original code, `repytah` extracts structural repetitions within a data stream and leverages their relationships to each other to build the aligned hierarchies representation. In addition to translating the code from MATLAB by cross-referencing the desired output of the package with the output of the original code, `repytah` improves on the original code, streamlining various computations and further modularizing functions. Additionally, this package provides more complete documentation, examples, and test files than the original code. 

# Functionality

There are four modules in the `repytah` Python package that work together to form the aligned hierarchies: `transform`, `search`, `assemble`, and `utilities`:

 - Functions in the `transform` module transform matrix inputs into different forms, either from lists of indices into matrices and vice versa. 
 - The `search` module finds and records information about repeated structures, represented as diagonals in a song’s thresholded self-dissimilarity matrix. 
 - Once found, these repeated structures are later transformed and assembled into the aligned hierarchies using the `assemble` module, which finds the essential structure components from the repeated structures found with the `search` module, and then uses those essential structure components to build the aligned hierarchies. 
 - Lastly, the `utilities` module contains functions that are frequently called by functions in the other three modules. 

Each module has an associated Jupyter notebook file that summarizes the module’s functions. There are also test files for each module. 

Additionally, the package includes `example.py` which runs a complete example building aligned hierarchies from a single example CSV file. This example CSV file has chroma features for the score of Chopin's Mazurka Op. 6, No. 1. 


# Acknowledgements
`repytah` has been partially from Smith College's Summer Undergraduate Research Fellowship (SURF) in 2019 - 2022 and by Smith College's CFCD funding mechanism. Additionally, as Kinnaird is the Clare Boothe Luce Assistant Professor of Computer Science and Statistical & Data Sciences at Smith College, this work has also been partially supported by Henry Luce Foundation's Clare Boothe Luce Program. Additionally, as `repytah` was developed in the TInKER lab (Katherine M. Kinnaird, Founder and PI), we would like to acknowlege other members of the TInKER lab, including Tasha Adler, for being part of our lab and for listening to presentations about earlier versions of this work. 

Finally, we would like to acknowledge and give thanks to Brian McFee and the [`librosa`](https://github.com/librosa) team. We significantly referenced the Python package [`librosa`](https://github.com/librosa) in our development process. 

# References

