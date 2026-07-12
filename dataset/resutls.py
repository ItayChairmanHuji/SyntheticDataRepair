from dataclasses import dataclass


@dataclass
class Results:
    deletion_ratio: float = -1
    marginals_error: float = -1
    marginals_distance: float = -1
    tvd: float = -1
    marginals_error_private: float = -1
    marginals_distance_private: float = -1
    num_of_violations: int = -1
