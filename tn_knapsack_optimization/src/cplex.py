from qiskit_optimization import QuadraticProgram
from qiskit_optimization import QiskitOptimizationError
from qiskit_optimization.algorithms import CplexOptimizer
import numpy as np
from time import time

#############################
#  KP solver: CPlex 4 BLIP  #
#############################

def cplex_kp_solver(kp_profits, kp_weights, max_capacity, n_items = None):
    """
    Solves the Knapsack Problem (KP) via CPlex 4 BLIP.

    """
    # Checking the input
    if n_items is None:
        n_items = len(kp_profits)
    assert n_items == len(kp_profits)
    assert n_items == len(kp_weights)
    assert max_capacity > 0

    
    # Creating the ILP as a Quadratic Program for Qiskit CPlex compatibility
    ## Creating a Quadratic Program instance (QP)
    model_name = "_".join([
        "KP",
        f"{n_items:04d}",
        f"{max_capacity:05d}",
        ])
    qp_model = QuadraticProgram(model_name)

    ## Adding the optimization variables 
    for jj in range(n_items):
        qp_model.binary_var(f'x_{jj}')
        
    ## Defining the quadratic cost function (only linear terms in this case)
    linear_term_dict = {f'x_{jj}': pj for jj, pj in enumerate(kp_profits)}
    qp_model.maximize(linear=linear_term_dict)
        
    ## Adding the linear inequality constraint (max capacity limit)
    lhs_ineq_const = {f'x_{jj}': wj for jj, wj in enumerate(kp_weights)}
    cname = f"The total weight of the knapsack must not exceed {max_capacity}"
    qp_model.linear_constraint(
        linear=lhs_ineq_const,
        sense="<=",
        rhs=max_capacity,
        name=cname
        )
    print("===================================")
    print("Knapsack problem instance as a BLIP")
    print("===================================")
    print(qp_model.prettyprint())
    print("===================================")
    print("===================================")
    print("\n\n")

    # Solving the BILP via CPlex
    ## Solver params
    number_of_threads = 1
    timelimit = 1e+75
    cplex_runtime = 0
    cplex_cost = 0
    cplex_solution = np.array([])
    cplex_status = ""
    cplex_correlation_matrix = np.array([])
    weight = 0

    ## Checking block
    if CplexOptimizer.is_cplex_installed():
        optimizer = CplexOptimizer(
            disp=False,
            cplex_parameters={
                'threads': number_of_threads,
                'randomseed': 0,
                'timelimit': timelimit
                }
            )
    else:
        raise Exception("CPlex classical optimizer "
                        "is not available on this machine.")
        
    ## Solving the QP

    try:
        st_time = time()
        cplex_result = optimizer.solve(qp_model)
        et_time = time()
        cplex_runtime = et_time - st_time
        cplex_cost = cplex_result.fval
        cplex_solution = cplex_result.x
        cplex_status = cplex_result.status.name
        cplex_correlation_matrix = cplex_result.get_correlations()
        weight = int(np.sum([wj * xj for wj, xj in zip(kp_weights, cplex_solution)]))
    except QiskitOptimizationError:
        status = " ".join([
            "The instantiated quadratic program",
            "is incompatible with the DOcplex optimizer."
            ])
        print(status)

    return {'combo': cplex_solution, 'profit': cplex_cost, 'cost': weight, 'runtime': cplex_runtime, 'status':cplex_status, 'corr': cplex_correlation_matrix}
