from denial_constraints import DenialConstraint, DenialConstraints, Predicate, Side


def test_side_value_index_is_none() -> None:
    side = Side.from_string("'Low'")

    assert isinstance(side.attr, str)
    assert side.index is None
    assert side.is_value is True


def test_side_tuple_index_is_int() -> None:
    side = Side.from_string("t2.education")

    assert isinstance(side.attr, str)
    assert isinstance(side.index, int)
    assert side.index == 2
    assert side.is_value is False


def test_predicate_fields_have_expected_types() -> None:
    predicate = Predicate.from_string("t1.A != t2.B")

    assert isinstance(predicate.left, Side)
    assert isinstance(predicate.right, Side)
    assert isinstance(predicate.opr, str)


def test_denial_constraint_fields_have_expected_types() -> None:
    dc = DenialConstraint.from_string("not(t1.A=t2.A&t1.B!=0)")

    assert isinstance(dc.predicates, list)
    assert all(isinstance(p, Predicate) for p in dc.predicates)


def test_denial_constraints_fields_have_expected_types(tmp_path) -> None:
    path = tmp_path / "dcs.txt"
    path.write_text("not(t1.A=t2.A)\nnot(t1.B!=0)", encoding="utf-8")

    dcs = DenialConstraints.from_dataset_name(path)

    assert isinstance(dcs.constraints, list)
    assert all(isinstance(dc, DenialConstraint) for dc in dcs.constraints)
