---
title: '`ah`: A Python Package for creating Aligned Hierarchies'
tags:
  - Python
  - Music Information Retrieval
  - Structure representations

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

# Abstract

AH is a python package that contributes to the efforts in the MIR community towards increased accessibility and reproducibility. This package is based on Katherine M. Kinnaird?s thesis $\textbf{\textit{Aligned Hierarchies for Sequential Data}$ and the accompanying MATLAB code. In Kinnaird's work, over 70 Matlab scripts build the Aligned Hierarchies. Improving upon Kinnaird's work, our package provides further accessibility and opportunity for reproducibility. Now written in python, ah creates aligned-hierarchies of music based data streams through finding and encoding repeated structures. The package is organized into four modules - utilities, search, transform and assemble - that work in tandem to extract and output the aligned hierarchies of music data. The package includes five vignettes that provide full detailed explanation of how each function works with examples and images. An overarching vignette is also provided with an example input of a Mazurka score that walks the user through the process and logic of the package. In addition to the framing and documentation of the package, ah includes unit tests for all functions and are organized by the modules. 


# Statement of need 



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
``` 

# Acknowledgements



# References

<!-- Format from https://joss.readthedocs.io/en/latest/submitting.html -->