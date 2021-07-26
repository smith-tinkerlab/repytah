import pkg_resources
import pandas as pd

def load_input():
    """Returns the mazurka score as a csv file for creating examples.
    """

    stream = pkg_resources.resource_stream(__name__, 'data/input.csv')
    return pd.read_csv(stream)
