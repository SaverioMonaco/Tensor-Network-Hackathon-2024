import numpy as np
from tqdm import trange


############################
#  KP solver: Brute-Force  #
############################

def toBitstrings(vector, bits:None | int =None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of bitstrings vectors, with 'bits' number of bits.

    """

    assert len(vector) > 0

    if bits is None:
        bits = np.ceil(np.log2(np.max(vector))).astype(int)
    else:
        assert bits > 0
        assert bits >= np.ceil(np.log2(np.max(vector)))

    result = np.array(vector)
    result = (((result[:,None] & (1 << np.arange(bits)))) > 0).astype(bool)  # little-endian
    
    return result

# Brute-force solver given some combinations
def kp_brute_force_combo(profit: np.ndarray, weight: np.ndarray, capacity: int, combinations: np.ndarray):
    """
    Computes the profit and weight of a given combination of items and returns the optimal one.

    """

    assert profit.shape[0] == weight.shape[0]
    assert profit.shape[0] == combinations.shape[-1]
    assert profit.shape[0] > 0
    assert capacity > 0

    max_profit = 0
    max_weight = 0
    max_combination = np.zeros(profit.shape[0])

    # Computing the costs and profits
    costs = combinations @ weight
    mask = costs <= capacity
    # get the profits of the combinations that satisfy the capacity constraint
    profits = (combinations[mask] @ profit)
    
    if profits.size > 0:
        max_profit = np.max(profits)
        max_weight = np.max(costs[mask])
        max_combination = combinations[mask][np.argmax(profits)]
    
    return {'profit': max_profit, 'cost': max_weight, 'combo': max_combination}


# Brute-force solver
def kp_brute_force(profit: np.ndarray, weight: np.ndarray, capacity: int, max_ram: int = 1E+9):
    n = profit.shape[0]
    assert n == weight.shape[0]
    assert n > 0
    assert capacity > 0
    assert max_ram > 0

    max_samples = int(max_ram / 2 // (n+8))
    n_steps = 2**n // max_samples + 1

    max_profits = np.zeros(n_steps, dtype=int)
    max_weights = np.zeros(n_steps, dtype=int)
    max_combinations = np.zeros((n_steps, n), dtype=bool)

    for i, slice_idx in enumerate(trange(0, 2**n, max_samples)):
        if slice_idx + max_samples < 2**n:
            combos = np.arange(slice_idx, slice_idx+max_samples)
            all_combinations = toBitstrings(combos, n)
        else:
            combos = np.arange(slice_idx, 2**n)
            all_combinations = toBitstrings(combos, n)
        max_profits[i], max_weights[i], max_combinations[i] = kp_brute_force_combo(profit, weight, capacity, all_combinations).values()
        del all_combinations, combos

    solution_id = np.argmax(max_profits)
    
    return {'profit': max_profits[solution_id], 'cost': max_weights[solution_id], 'combo': np.nonzero(max_combinations[solution_id])[0]}
    