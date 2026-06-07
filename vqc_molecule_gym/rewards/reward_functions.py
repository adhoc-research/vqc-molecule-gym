from vqc_molecule_gym.schemas.result import CircuitMetrics
from vqc_molecule_gym.schemas.task import Constraints

REWARD_VERSION = "reward_v1.1.0"
REWARD_WEIGHTS = {
    "energy_error": 0.90,
    "chemical_accuracy": 0.08,
    "depth": 0.01,
    "shots": 0.005,
    "compactness": 0.005,
}


def reward_v1(
    *,
    valid: bool,
    energy_error_mha: float | None,
    metrics: CircuitMetrics | None,
    shots: int,
    constraints: Constraints,
) -> tuple[float, dict[str, float]]:
    """Dense reward dominated by absolute energy error.

    The main term is a smooth bounded transform of the error in units of the
    task's chemical-accuracy threshold. It is strictly decreasing with error:
    0 mHa maps to +1, the chemical-accuracy threshold maps to 0, and very large
    errors asymptote to -1. Resource terms are intentionally tiny tie-breakers;
    they should not delay learning from meaningful energy-error reductions.
    """
    if not valid or energy_error_mha is None or metrics is None:
        return -1.0, {"validity": -1.0}

    error = max(float(energy_error_mha), 0.0)
    chemical_accuracy_mha = constraints.chemical_accuracy_mha
    normalized_error = error / chemical_accuracy_mha
    energy_error_score = 1.0 - (2.0 * normalized_error / (1.0 + normalized_error))

    depth_score = 1.0 - min(metrics.depth / constraints.max_depth, 1.0)
    shot_score = 1.0 - min(shots / constraints.max_shots, 1.0)
    compactness = 1.0 - min(metrics.num_operators / constraints.max_operators, 1.0)
    chem_bonus = 1.0 if error <= chemical_accuracy_mha else 0.0

    components = {
        "energy_error": REWARD_WEIGHTS["energy_error"] * energy_error_score,
        "chemical_accuracy": REWARD_WEIGHTS["chemical_accuracy"] * chem_bonus,
        "depth": REWARD_WEIGHTS["depth"] * depth_score,
        "shots": REWARD_WEIGHTS["shots"] * shot_score,
        "compactness": REWARD_WEIGHTS["compactness"] * compactness,
    }
    return sum(components.values()), components
