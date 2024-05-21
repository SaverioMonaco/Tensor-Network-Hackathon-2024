"""Computation of the hamiltonian matrix

We obtain the following classical many-body Ising-like Hamiltonian:

H = Cte * 1 + h_k \sigma_k + j_kk' \sigma_k\sigma_k'
"""

# %%
import numpy as np 
from typing import List
from numbers import Number

def get_Q(weights : List[int], profits : List[int], C : int, l : Number):
    assert len(weights) == len(profits)
    
    N = len(weights) 
    M = int(np.floor(np.log2(C)))

    nq = N + M + 1

    Q = np.zeros((nq, nq))

    K = (C + 1 - 2**M)
    # Diagonal terms
    for n in range(N):
        Q[n,n] += l*(weights[n]**2) - profits[n]

    for m in range(M):
        Q[N+m, N+m] += l*(2**(2*m))

    Q[N+M, N+M] += l*K**2

    # Off diagonal terms
    # xj xj' 
    for n1 in range(N):
        for n2 in range(n1):
            Q[n1, n2] += 2*l*weights[n1]*weights[n2]
            Q[n2, n1]  = Q[n1, n2]
    # ba ba'
    for a1 in range(M):
        for a2 in range(a1):
            Q[N+a1, N+a2] += l*2**(a1 + a2 + 1)
            Q[N+a2, N+a1]  = Q[N+a1, N+a2]
    # xj ba 
    for n in range(N):
        for a in range(M):
            Q[n, N+a] += - l * weights[n] * (2**(a + 1))
            Q[N+a, n]  = Q[n, N+a]
    # xj bM
    for n in range(N):
        Q[n, N+M] += - 2*l*K * weights[n]
        Q[N+M, n]  = Q[n, N+M]
    # ba bM
    for a in range(M):
        Q[N+a, N+M] += 2*l*K * (2**a)
        Q[N+M, N+a]  = Q[N+a, N+M]

    return Q
