import numpy as np


def fermionic_hamiltonian_matrix(
    h1: np.ndarray,
    h2: np.ndarray,
    nuclear_repulsion: float,
) -> np.ndarray:
    n_qubits = h1.shape[0]
    dim = 1 << n_qubits
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    matrix += nuclear_repulsion * np.eye(dim, dtype=np.complex128)

    for p in range(n_qubits):
        for q in range(n_qubits):
            coeff = h1[p, q]
            if abs(coeff) > 1e-12:
                _add_one_body(matrix, coeff, p, q, n_qubits)

    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    coeff = 0.5 * h2[p, q, r, s]
                    if abs(coeff) > 1e-12:
                        _add_two_body(matrix, coeff, p, q, r, s, n_qubits)
    return matrix


def fixed_electron_subspace_indices(n_qubits: int, n_electrons: int) -> list[int]:
    return [idx for idx in range(1 << n_qubits) if idx.bit_count() == n_electrons]


def exact_ground_energy(matrix: np.ndarray, n_qubits: int, n_electrons: int) -> float:
    indices = fixed_electron_subspace_indices(n_qubits, n_electrons)
    block = matrix[np.ix_(indices, indices)]
    return float(np.linalg.eigvalsh(block)[0].real)


def _add_one_body(matrix: np.ndarray, coeff: complex, p: int, q: int, n_qubits: int) -> None:
    for state in range(1 << n_qubits):
        applied = _annihilate(state, q)
        if applied is None:
            continue
        phase1, next_state = applied
        applied = _create(next_state, p)
        if applied is None:
            continue
        phase2, final_state = applied
        matrix[final_state, state] += coeff * phase1 * phase2


def _add_two_body(
    matrix: np.ndarray,
    coeff: complex,
    p: int,
    q: int,
    r: int,
    s: int,
    n_qubits: int,
) -> None:
    for state in range(1 << n_qubits):
        applied = _annihilate(state, s)
        if applied is None:
            continue
        phase, next_state = applied
        for op, orbital in ((_annihilate, r), (_create, q), (_create, p)):
            applied = op(next_state, orbital)
            if applied is None:
                break
            op_phase, next_state = applied
            phase *= op_phase
        else:
            matrix[next_state, state] += coeff * phase


def _annihilate(state: int, orbital: int) -> tuple[int, int] | None:
    if not (state >> orbital) & 1:
        return None
    return _parity_phase(state, orbital), state & ~(1 << orbital)


def _create(state: int, orbital: int) -> tuple[int, int] | None:
    if (state >> orbital) & 1:
        return None
    return _parity_phase(state, orbital), state | (1 << orbital)


def _parity_phase(state: int, orbital: int) -> int:
    lower = state & ((1 << orbital) - 1)
    return -1 if lower.bit_count() % 2 else 1
