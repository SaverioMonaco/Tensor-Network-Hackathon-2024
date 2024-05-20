"""Load the data"""

# %% 
import numpy as np 
from typing import List, Literal 
import os 

def read(file : str):
    """Read the information from a single Knapsack problem file

    The file is structured as follows:
    row | 
    0   | Number of total x  |                        |
    1   | index (0)          | profit of the first x  | weight of the first x 
    2   | index (1)          | profit of the second x | weight of the second x 
    ... |                    |                        | 
    [-1]| Total capacity     |                        | 

    Parameters
    ----------
    file : str
        File to read the problem from the txt file

    Returns
    -------
    int
        Max Capacity C
    List[int] 
        List of profits p
    List[int]
        List of weights w
    """
    # Loading an instance from the KP instances folder
    profits = []
    weights = []
    with open(file, 'r') as f:
        n_items = int(f.readline())
        for jj in range(n_items):
            _, pj, wj = f.readline().split()
            profits.append(pj)
            weights.append(wj)
        max_capacity = int(f.readline())
    profits = np.array(profits, dtype=int)
    weights = np.array(weights, dtype=int)

    return max_capacity, profits, weights

def load(dataset : Literal['small', 'medium', 'large'] = 'small', path : str = '../kp_instances'):
    """Load multiple problems from a folder into an array of dictionaries (one for each problem)

    Parameters
    ----------
    dataset : Literal['small', 'medium', 'large']
        Name of the folder 
    path : str
        Name of the path containing the folders

    Returns
    -------
    List[dict]
        Set containing all the problems        
    """ 
    full_path = f'{path}/{dataset}'
    files = os.listdir(full_path)

    # Check on files, if they begin with kp we are almost 
    # sure they are the correct files
    files = [f'{full_path}/{file}' for file in files if file[:2] == 'kp']
    
    set : list = []
    for file in files:
        C, profits, weights = read(file)
        set.append({})
        set[-1]['C'] = C
        assert len(weights) == len(profits)
        set[-1]['n'] = len(weights)
        set[-1]['weights'] = list(weights)
        set[-1]['profits'] = list(profits)
        
    return set
