from dataclasses import dataclass
from pathlib import Path
from typing import Self

from denial_constraints.denial_constraint import DenialConstraint


@dataclass
class DenialConstraints:
    BASE_PATH = Path("resources/constraints/")

    constraints: list[DenialConstraint]

    @classmethod
    def from_dataset_name(cls, dataset_name: str | Path) -> Self:
        path = Path(dataset_name)
        if path.exists() or path.suffix:
            return cls.from_file(path)
        return cls.from_file(cls.BASE_PATH / f"{dataset_name}.txt")

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        constraints = [DenialConstraint.from_string(line) for line in lines if line.strip()]
        return cls(constraints=constraints)

    @property
    def attrs(self) -> set[str]:
        return set().union(*(dc.attrs for dc in self.constraints))

    def to_string(self) -> str:
        return "\n".join(dc.to_string() for dc in self.constraints)
