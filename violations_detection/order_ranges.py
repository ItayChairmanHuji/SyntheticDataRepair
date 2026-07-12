import numpy as np


def order_range(operator: str, value, values: np.ndarray) -> tuple[int, int]:
    match operator:
        case ">":
            return 0, int(np.searchsorted(values, value, side="left"))
        case ">=":
            return 0, int(np.searchsorted(values, value, side="right"))
        case "<":
            return int(np.searchsorted(values, value, side="right")), len(values)
        case "<=":
            return int(np.searchsorted(values, value, side="left")), len(values)
    raise ValueError(f"Unsupported order operator: {operator}")
