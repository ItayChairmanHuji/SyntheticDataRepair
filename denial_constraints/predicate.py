import re
from dataclasses import dataclass
from typing import Self

from denial_constraints.side import Side


@dataclass
class Predicate:
    left: Side
    opr: str
    right: Side

    @classmethod
    def from_string(cls, raw: str) -> Self:
        raw = raw.strip()

        pattern = r"(.+?)\s*(==|!=|<=|>=|=|<|>)\s*(.+)"
        match = re.fullmatch(pattern, raw)

        if not match:
            raise ValueError(f"Invalid predicate: {raw}")

        right_raw = match.group(3).strip()
        if right_raw.startswith(("=", "!", "<", ">")):
            raise ValueError(f"Invalid predicate: {raw}")

        left = Side.from_string(match.group(1))
        opr = match.group(2)
        right = Side.from_string(right_raw)

        if left.is_value:
            raise ValueError(f"Invalid predicate left side: {raw}")

        return cls(left=left, opr=opr, right=right)

    @property
    def is_unary(self) -> bool:
        return self.right.is_value

    @property
    def attrs(self) -> set[str]:
        attrs = set()
        if not self.left.is_value:
            attrs.add(self.left.attr)
        if not self.right.is_value:
            attrs.add(self.right.attr)
        return attrs

    def to_string(self) -> str:
        return f"{self.left.to_string()} {self.opr} {self.right.to_string()}"
