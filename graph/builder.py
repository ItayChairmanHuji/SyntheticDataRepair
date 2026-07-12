from __future__ import annotations

from .graph import Graph
from .initializer import GraphBuilder
from .types import ViolationSetLike


class ConflictGraphBuilder:
    @staticmethod
    def build(n: int, violation_set: ViolationSetLike) -> Graph:
        return GraphBuilder(n, violation_set).graph
