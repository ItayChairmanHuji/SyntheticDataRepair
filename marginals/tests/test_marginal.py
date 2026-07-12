import pandas as pd

from marginals.marginal import Marginal


def test_mask_matches_numeric_values_with_different_representations() -> None:
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [12.0, 13.0, 13.0],
        }
    )
    marginal = Marginal(["int_col", "float_col"], [2.0, "13.0"], 0.25)

    assert marginal.mask(df).tolist() == [False, True, False]


def test_mask_keeps_string_matching_for_categorical_values() -> None:
    df = pd.DataFrame(
        {
            "country": [" Puerto-Rico", "Puerto-Rico", " Canada"],
            "gender": ["M", "M", "F"],
        }
    )
    marginal = Marginal(["country", "gender"], [" Puerto-Rico", "M"], 0.25)

    assert marginal.mask(df).tolist() == [True, False, False]


def test_mask_compares_numeric_like_string_columns_numerically() -> None:
    df = pd.DataFrame({"x": ["1", "2", "3"]})
    marginal = Marginal(["x"], [2.0], 0.25)

    assert marginal.mask(df).tolist() == [False, True, False]
