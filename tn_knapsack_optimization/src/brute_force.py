import numpy as np
from tqdm import trange
from multiprocessing import Pool, cpu_count
from functools import partial

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
        max_index = np.argmax(profits)
        max_profit = profits[max_index]
        max_weight = costs[mask][max_index]
        max_combination = combinations[mask][max_index]
    
    return {'profit': max_profit, 'cost': max_weight, 'combo': max_combination}


# Brute-force solver
def kp_brute_force(profit: np.ndarray, weight: np.ndarray, capacity: int, max_ram: int = 6E+9):
    """
    Computes the profit and weight of all the possible combinations of items and returns the optimal one.

    Parameters
    ----------
    profit : np.ndarray
        The profit of each item.
    weight : np.ndarray
        The weight of each item.
    capacity : int
        The maximum capacity of the knapsack.
    max_ram : int
        The maximum RAM that the solver can use in bytes.
        This is not a strict limit, but a guideline to split the computation in smaller chunks.

    Returns
    -------
    dict
        A dictionary with the following keys:
            - 'profit': the maximum profit.
            - 'cost': the weight of the items that achieve the maximum profit.
            - 'combo': the combination of items that achieve the maximum profit.

    """
    n = profit.shape[0]
    assert n == weight.shape[0]
    assert n > 0
    assert capacity > 0
    assert max_ram > 0

    n_threads = cpu_count()
    max_samples = int(max_ram / 2 // (n+8))
    n_steps = 2**n // max_samples + 1

    max_profits = np.zeros(n_steps, dtype=int)
    max_weights = np.zeros(n_steps, dtype=int)
    max_combinations = np.zeros((n_steps, n), dtype=bool)

    for i, slice_idx in enumerate(trange(0, 2**n, n_threads*max_samples, disable=False)):

        if slice_idx + n_threads*max_samples < 2**n:
            remaining_threads = n_threads
            pool = Pool(n_threads)
            results = pool.map(partial(kp_brute_force_combo, profit, weight, capacity), [slice(slice_idx+j*max_samples, slice_idx+(j+1)*max_samples) for j in range(n_threads)])
        else:
            remaining_threads = np.ceil((2**n - slice_idx) / max_samples).astype(int)
            slices = [slice(slice_idx+j*max_samples, (slice_idx+(j+1)*max_samples if j != remaining_threads -1 else 2**n)) for j in range(remaining_threads)]
            pool = Pool(remaining_threads)
            results = pool.map(partial(kp_brute_force_combo, profit, weight, capacity), [ slice_j for slice_j in slices])

        pool.close()
        pool.join()

        max_profits[i*n_threads:i*n_threads+remaining_threads] = [result['profit'] for result in results]
        max_weights[i*n_threads:i*n_threads+remaining_threads] = [result['cost'] for result in results]
        max_combinations[i*n_threads:i*n_threads+remaining_threads] = [result['combo'] for result in results]

    solution_id = np.argmax(max_profits)
    
    return {'profit': max_profits[solution_id], 'cost': max_weights[solution_id], 'combo': np.nonzero(max_combinations[solution_id])[0]}
    