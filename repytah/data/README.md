# Data for Examples with <img alt="repytah" src="branding/repytah_logo.png" height="100">

In this directory, there are three CSV files that can be used to test the installation for `repytah`. The three CSV files are for three of Chopin's Mazurkas and they are based on KERN data from the [kern Scores data base](https://kern.humdrum.org/cgi-bin/browse?l=users/craig/classical/chopin/mazurka) [@S2005]. These files were processed from the kern format to Chroma features (or the twelve Western tones: {C, C#, D, D#, etc}) using [`music21`](https://web.mit.edu/music21/). For each beat in the score, we have a vector of 12 entries, encoding how much of each note we have. This representation removes all octave information, meaning that a high C and a middle C sounding in the same beat would each contribute "1" to the entry assocated to "C."

The three Mazurkas are:

* Op 6, No 1
* Op 30, No 1
* Op 30, No 3

## Preprocessing data for `repytah`

The main purpose of `repytah` is to build the aligned hierarchies representation for sequential data from a self-dissimilarity matrix representing that data. This means that one must select both the unit of time (seconds, miliseconds, beats) as well as the dissimilarity measure used to compare those units. 

In `example.py`, we create the aligned hierarchies for the score of Chopin's Mazurka Op. 6, No. 1. We have features by beat and enforce a notion of proximity by "shingling" the features, that is for each beat we consider the features for that beat plus `s` additional beats [@CS2007; @CS2006NEAR; CS2006SEQ]. We then create the SDM for the shingles using the cosine-dissimilarity measure. 


## References

@article{S2005,
	author={C.S. Sapp},
	title={Online Database of Scores in the Humdrum File Format},
	journal=ismir06,
	pages={664-665},
	year= 2005
}


@article{CS2007,
	author={M. Casey, and M. Slaney},
	title={Fast Recognition of Remixed Audio},
	journal=assp7,
	pages={IV-1425 - IV-1428},
	year={2007},
}

@article{CS2006NEAR,
	author={M. Casey, and M. Slaney},
	title={Song Intersection by Approximate Nearest Neighbor Search},
	journal=ismir07,
	pages={144-149},
	year=2006,
}

@article{CS2006SEQ,
	author={M. Casey, and M. Slaney},
	title={The importance of sequences in music similarity},
	journal=assp6,
	pages={V-5 - V-8},
	year=2006
}