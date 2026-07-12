import pytest

from denial_constraints import DenialConstraint, DenialConstraints, Predicate, Side

VALID_CONSTRAINTS = [
    "not(t1.education_num != t2.education_num&t1.education=t2.education)",
    "not(t1.education_num = t2.education_num&t1.education!=t2.education)",
    "not(t1.capital_gain < t2.capital_gain&t1.capital_loss<t2.capital_loss)",
    "not(t1.AGEP=0&t2.AGEP=0&t1.SCHL=0&t2.SCHL!=0)",
    "not(t1.AGEP=0&t2.AGEP=0&t1.WAGP=0&t2.WAGP!=0)",
    "not(t1.AGEP=0&t2.AGEP=0&t1.MAR=5&t2.MAR!=5)",
    "not(t1.CIT=t2.CIT&t1.NATIVITY!=t2.NATIVITY)",
    "not(t1.ST=t2.ST&t1.DIVISION!=t2.DIVISION)",
    "not(t1.ScoreText='Low'&t2.ScoreText='High'&t1.DecileScore>t2.DecileScore)",
    "not(t1.Scale_ID=t2.Scale_ID&t1.DisplayText!=t2.DisplayText)",
    "not(t1.RecSupervisionLevelText='Low'&t2.RecSupervisionLevelText='High'&t1.RecSupervisionLevel>t2.RecSupervisionLevel)",
    "not(t1.DisplayText=t2.DisplayText&t1.Scale_ID!=t2.Scale_ID)",
    "not(t1.RecSupervisionLevelText=t2.RecSupervisionLevelText&t1.RecSupervisionLevel!=t2.RecSupervisionLevel)",
    "not(t1.State=t2.State&t1.HasChild=t2.HasChild&t1.ChildExemp!=t2.ChildExemp)",
    "not(t1.State=t2.State&t1.MaritalStatus=t2.MaritalStatus&t1.SingleExemp!=t2.SingleExemp)",
    "not(t1.State=t2.State&t1.Salary>t2.Salary&t1.Rate<t2.Rate)",
    "not(t1.State=t2.State&t1.Salary<t2.Salary&t1.Rate>t2.Rate)",
]


@pytest.mark.parametrize("raw", VALID_CONSTRAINTS)
def test_parse_valid_denial_constraint(raw: str) -> None:
    dc = DenialConstraint.from_string(raw)

    assert isinstance(dc, DenialConstraint)
    assert len(dc.predicates) >= 1

    for predicate in dc.predicates:
        assert isinstance(predicate, Predicate)
        assert isinstance(predicate.left, Side)
        assert isinstance(predicate.right, Side)
        assert predicate.opr in {"=", "!=", "<", ">", "<=", ">="}


def test_parse_all_constraints_from_file(tmp_path) -> None:
    path = tmp_path / "constraints.txt"
    path.write_text("\n".join(VALID_CONSTRAINTS), encoding="utf-8")

    dcs = DenialConstraints.from_dataset_name(path)

    assert isinstance(dcs, DenialConstraints)
    assert len(dcs.constraints) == len(VALID_CONSTRAINTS)


def test_binary_predicate_types() -> None:
    predicate = Predicate.from_string("t1.education_num != t2.education_num")

    assert predicate.left.attr == "education_num"
    assert predicate.left.index == 1
    assert predicate.left.is_value is False

    assert predicate.opr == "!="

    assert predicate.right.attr == "education_num"
    assert predicate.right.index == 2
    assert predicate.right.is_value is False


def test_unary_numeric_predicate_types() -> None:
    predicate = Predicate.from_string("t1.AGEP = 0")

    assert predicate.left.attr == "AGEP"
    assert predicate.left.index == 1
    assert predicate.left.is_value is False

    assert predicate.opr == "="

    assert predicate.right.attr == "0"
    assert predicate.right.index is None
    assert predicate.right.is_value is True


def test_unary_string_predicate_types() -> None:
    predicate = Predicate.from_string("t1.ScoreText = 'Low'")

    assert predicate.left.attr == "ScoreText"
    assert predicate.left.index == 1
    assert predicate.left.is_value is False

    assert predicate.opr == "="

    assert predicate.right.attr == "Low"
    assert predicate.right.index is None
    assert predicate.right.is_value is True


def test_attrs_excludes_literal_values() -> None:
    dc = DenialConstraint.from_string("not(t1.ScoreText='Low'&t2.ScoreText='High'&t1.DecileScore>t2.DecileScore)")

    assert dc.attrs == {"ScoreText", "DecileScore"}


def test_to_string_roundtrip_parseable() -> None:
    original = "not(t1.AGEP=0&t2.AGEP=0&t1.SCHL=0&t2.SCHL!=0)"

    dc = DenialConstraint.from_string(original)
    serialized = dc.to_string()
    parsed_again = DenialConstraint.from_string(serialized)

    assert parsed_again.to_string() == serialized


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not()",
        "t1.",
        "t1.A === t2.A",
        "not(t1.A=t2.A&t1.B)",
        "not(foo)",
    ],
)
def test_invalid_constraints_raise_value_error(raw: str) -> None:
    with pytest.raises(ValueError):
        DenialConstraint.from_string(raw)
