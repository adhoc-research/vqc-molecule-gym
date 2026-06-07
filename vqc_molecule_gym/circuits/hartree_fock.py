def hartree_fock_occupied_qubits(n_electrons: int) -> list[int]:
    return list(range(n_electrons))


def basis_labels_little_endian(n_qubits: int) -> list[list[int]]:
    return [[(idx >> bit) & 1 for bit in range(n_qubits)] for idx in range(1 << n_qubits)]
