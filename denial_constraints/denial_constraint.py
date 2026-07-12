from typing import Self

from attr import dataclass

from denial_constraints.predicate import Predicate


@dataclass
class DenialConstraint:
    predicates: list[Predicate]

    @classmethod
    def from_string(cls, raw: str) -> Self:
        raw = raw.strip()
        raw = cls._remove_outer_not(raw)
        if not raw:
            raise ValueError("Invalid denial constraint: empty constraint")
        predicates = [Predicate.from_string(part) for part in raw.split("&") if part.strip()]
        if not predicates:
            raise ValueError("Invalid denial constraint: empty constraint")
        return cls(predicates=predicates)

    @classmethod
    def _remove_outer_not(cls, raw: str) -> str:
        return raw[4:-1] if raw.startswith("not(") and raw.endswith(")") else raw

    @property
    def attrs(self) -> set[str]:
        return set().union(*(p.attrs for p in self.predicates))

    def to_string(self) -> str:
        return "not(" + " & ".join(p.to_string() for p in self.predicates) + ")"
