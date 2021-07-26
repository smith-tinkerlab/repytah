#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
input.py 

This module contains the load_csv_input function used to load the example data.
"""

import pkg_resources
import pandas as pd

def load_csv_input(input):
    """
    Reads in a csv input file.

    Args
    ----
    input : str
        Name of .csv file to be processed. Contains features across time steps 
        to be analyzed, for example chroma features
    """

    stream = pkg_resources.resource_stream(__name__, input)
    return pd.read_csv(stream)
