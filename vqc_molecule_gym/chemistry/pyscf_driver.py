"""Deprecated compatibility shim for the former direct PySCF driver.

Molecular Hamiltonian generation is now routed through CUDA-QX Solvers. PySCF
remains an environment dependency because CUDA-QX's local molecule backend uses
it internally, but project code should import from ``cudaqx_driver`` directly.
"""

from vqc_molecule_gym.chemistry.cudaqx_driver import MolecularData, build_molecular_data

__all__ = ["MolecularData", "build_molecular_data"]
