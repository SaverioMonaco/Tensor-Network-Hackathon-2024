from qiskit.result import QuasiDistribution

from qiskit_algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA

from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src import ising
from src import qubo


def qaoa(values, weights, C, reps=2):

    # get the qubo matrix
    LAMBDA = max(values)*1.1
    Q = qubo.get_Q(weights, values, C, LAMBDA)

    # embed the qubo matrix into an hamiltonian
    h, J, offset = ising.qubo_to_hamiltonian(Q)

    H = ising.construct_quantum_hamiltonian_qiskit(h, J, offset)

    sampler = Sampler()
    algorithm_globals.random_seed = 10598

    optimizer = COBYLA()
    qaoa = QAOA(sampler, optimizer, reps=reps)

    result = qaoa.compute_minimum_eigenvalue(H)

    return result