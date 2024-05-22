
def construct_quantum_hamiltonian_scipy(h, J, Cte):
    n = len(h)

    # ⚠️ THIS USES BIG ENDIANNESS :)

    # as the operator is diagonal, we just calculate the diagonal elements
    I = np.array([1,1], dtype=np.int32)

    def create_operator_diag(ops : List[NDArray], indices : List[int],  n_qubits : int):
        result = 1

        assert len(ops) == len(indices)
        if np.any([op.ndim > 1 for op in ops]):
            raise ValueError("Ops must be diagonal")

        for i in range(n_qubits):
            if i in indices:
                op = ops[indices.index(i)]
            else:
                op = I

            result = np.kron(result, op)

        return result
        
    H = np.ones((2**n), dtype=np.float32) * Cte

    Z = np.array([1,-1], dtype=np.int32)
    for i in range(n):
        H += h[i]*create_operator_diag([Z], [i], n)
    for (i, j), J_ij in J.items():
        H += J_ij*create_operator_diag([Z,Z], [i,j], n)
    return H 