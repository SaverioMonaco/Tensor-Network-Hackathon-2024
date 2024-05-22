import numpy as np
from tqdm import trange
from multiprocessing import Pool, cpu_count
from functools import partial

############################
#  KP solver: Brute-Force  #
############################

def toBitstrings(vector, bits:None | int =None, big_endian:bool=True):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of bitstrings vectors, with 'bits' number of bits.

    """

    if bits is None:
        bits = np.floor(np.log2(np.max(vector))).astype(int)+1
    else:
        assert bits > 0
        assert bits >= np.ceil(np.log2(np.max(vector)))

    result = np.array(vector)
    result = (((result[...,None] & (1 << np.arange(bits)))) > 0).astype(bool)
    if big_endian:
        result = result[...,::-1]  # big-endian
    
    return result

# Brute-force solver given some combinations
def kp_brute_force_combo(profit: np.ndarray, weight: np.ndarray, capacity: int, combinations: np.ndarray | slice):
    """
    Computes the profit and weight of a given combination of items and returns the optimal one.

    """

    if isinstance(combinations, slice):
        combinations = toBitstrings(np.arange(combinations.start, combinations.stop, combinations.step), bits=profit.shape[0])

    assert profit.shape[0] == weight.shape[0]
    assert profit.shape[0] == combinations.shape[-1]
    assert profit.shape[0] > 0
    assert capacity > 0


    max_profits = np.zeros((1,), dtype=int)
    max_weights = np.zeros((1,), dtype=int)
    max_combinations = np.zeros((1, profit.shape[0]), dtype=bool)

    # Computing the costs and profits
    costs = combinations @ weight
    mask = costs <= capacity

    # get the profits of the combinations that satisfy the capacity constraint
    profits = (combinations[mask] @ profit)
    
    if profits.size > 0:
        max_indices = np.flatnonzero(profits == np.max(profits))
        max_profits = profits[max_indices]
        max_weights = costs[mask][max_indices]
        max_combinations = combinations[mask][max_indices]
        if max_combinations.ndim < 2:
            max_combinations = np.expand_dims(max_combinations, 0)
    return {'profit': max_profits, 'cost': max_weights, 'combo': max_combinations}


# Brute-force solver
def kp_brute_force(profit: np.ndarray | list, weight: np.ndarray | list, capacity: int, max_ram: int = 6E+9):
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
    if isinstance(profit, list):
        profit = np.array(profit)
    if isinstance(weight, list):
        weight = np.array(weight)
    n = profit.shape[0]
    assert n == weight.shape[0]
    assert n > 0
    assert capacity > 0
    assert max_ram > 0

    n_threads = cpu_count()
    max_samples = int(max_ram / 2 // (n+8))//40
    n_steps = 2**n // max_samples + 1

    print(f"Number of threads: {n_threads}")
    print(f"Max samples: {max_samples}")
    print(f"Number of steps: {n_steps}")

    max_profits = []#np.zeros(n_steps, dtype=int)
    max_weights = []#np.zeros(n_steps, dtype=int)
    max_combinations = []#np.zeros((n_steps, n), dtype=bool)

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

        threads_profits = np.concatenate([result['profit'] for result in results])
        threads_weights = np.concatenate([result['cost'] for result in results])
        threads_combinations = np.concatenate([result['combo'] for result in results])
        threads_sols = np.flatnonzero(threads_profits == np.max(threads_profits))

        max_profits.extend(threads_profits[threads_sols])
        max_weights.extend(threads_weights[threads_sols])
        max_combinations.extend(threads_combinations[threads_sols])
        del threads_profits, threads_weights, threads_combinations, threads_sols, results
        '''
        max_profits[i*n_threads:i*n_threads+remaining_threads] = [result['profit'] for result in results]
        max_weights[i*n_threads:i*n_threads+remaining_threads] = [result['cost'] for result in results]
        max_combinations[i*n_threads:i*n_threads+remaining_threads] = [result['combo'] for result in results]
        '''
    max_profits = np.stack(max_profits)
    max_weights = np.stack(max_weights)
    max_combinations = np.stack(max_combinations)

    solution_id = np.flatnonzero(max_profits == np.max(max_profits))
    
    return {'profit': max_profits[solution_id], 'cost': max_weights[solution_id], 'combo': [np.nonzero(max_combinations[solution_id][i])[0] for i in range(len(solution_id))]}
    