import numpy as np

from qiskit.quantum_info import Pauli, SparsePauliOp

def qubo_to_hamiltonian(qubo):
    n = qubo.shape[0]

    h = {}
    J = {}

    # generate the h_k and J_k_k' values
    for k in range(n):
        h_k_sum = sum(qubo[k, i] for i in range(k + 1, n))
        h[k] = 0.5 * qubo[k, k] + 0.25 * h_k_sum
        for j in range(k + 1, n):
            J[(k, j)] = 0.25 * qubo[k, j]

    Q_kk = sum(qubo[k, k] for k in range(n))
    Q_kk_prime = sum(qubo[k, k_prime] for k in range(n) for k_prime in range(k + 1, n))
    Cte = 0.5 * Q_kk + 0.25 * Q_kk_prime

    return h, J, Cte

def construct_quantum_hamiltonian(h, J, Cte):
    n = len(h)
    pauli_list = []
    coeffs = []

    # Constant term
    pauli_list.append(Pauli('I' * n))
    coeffs.append(Cte)

    # Linear terms
    for i in range(n):
        z_pauli = 'I' * i + 'Z' + 'I' * (n - i - 1)
        pauli_list.append(Pauli(z_pauli))
        coeffs.append(h[i])

    # Quadratic terms
    for (i, j), J_ij in J.items():
        zz_pauli = 'I' * i + 'Z' + 'I' * (j - i - 1) + 'Z' + 'I' * (n - j - 1)
        pauli_list.append(Pauli(zz_pauli))
        coeffs.append(J_ij)

    H = SparsePauliOp(pauli_list, coeffs)
    return H



