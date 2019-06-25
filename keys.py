#-------------------------------------
# Libraries
#------------------------------------
import pandas as pd
import json 

#-------------------------------------
# Functions
#------------------------------------
def get_keys(path):
    """
    Given a path to json file, returns the keys
    Parameters
    ---------
    path: path with file name of json file
    Returns
    -------
    returns: dict of keys
    """
    with open(path) as f:
        return json.load(f)