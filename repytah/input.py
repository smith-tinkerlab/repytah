#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
input.py 

This module contains the load_input function used to load the example data.
"""

import pkg_resources
import pandas as pd

def load_input():
    """Returns the mazurka score as a csv file for creating examples.
    """

    stream = pkg_resources.resource_stream(__name__, 'data/input.csv')
    return pd.read_csv(stream)
