import re
from dataclasses import dataclass
from typing import Self


@dataclass
class Side:
    attr: str
    index: int | None
    is_value: bool = False

    @classmethod
    def from_string(cls, raw: str) -> Self:
        raw = raw.strip()

        if match := re.fullmatch(r"t(\d+)\.([A-Za-z_]\w*)", raw):
            return cls(
                index=int(match.group(1)),
                attr=match.group(2),
                is_value=False,
            )

        return cls(
            index=None,
            attr=raw.strip("'\""),
            is_value=True,
        )

    def to_string(self) -> str:
        if not self.is_value:
            return f"t{self.index}.{self.attr}"

        try:
            float(self.attr)
            return str(self.attr)
        except ValueError:
            return f"'{self.attr}'"
