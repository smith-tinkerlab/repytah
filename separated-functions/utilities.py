#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
utilities.py 

This script when imported as a module allows search.py, disassemble.py and 
assemble.py in the ah package to run smoothly. 

This file contains the following functions:
    
    * reconstruct_full_block - Creates a record of when pairs of repeated
    structures occur, from the first beat in the song to the end. Pairs of 
    repeated structures are marked with 1's. 

    
    * find_initial_repeats - Finds all diagonals present in thresh_mat, 
    removing each diagonal as it is found.
    
    * add_annotations - Adds annotations to each pair of repeated structures 
    according to their order of occurence. 
    
    * create_sdm - Creates a self-dissimilarity matrix; this matrix is found 
    by creating audio shingles from feature vectors, and finding cosine 
    distance between shingles. 
    
    * reformat - Transforms a binary matrix representation of when repeats 
    occur in a song into a list of repeated structures detailing the length
    and occurence of each repeat. 
     
"""

