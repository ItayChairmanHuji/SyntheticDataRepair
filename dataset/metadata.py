from dataclasses import dataclass


@dataclass
class Metadata:
    data_type: str
    generation_model: str
    dataset_name: str
    epsilon: float
    size: int
    synthetic_seed: int
    marginals_type: str  | None = None
    num_of_marginals: int | None = None
    marginals_seed: int | None = None
    alpha: float | None = None

    @property
    def id(self) -> str:
        parts = [self.data_type]
        if self.generation_model != "":
            parts.append(self.generation_model)
        parts.append(self.dataset_name)
        if self.epsilon != -1:
            parts.append(str(self.epsilon))
        if self.size != -1:
            parts.append(str(self.size))
        if self.synthetic_seed != -1:
            parts.append(str(self.synthetic_seed))
        if self.num_of_marginals is not None:
            parts.append(str(self.num_of_marginals))
        if self.marginals_seed is not None:
            parts.append(str(self.marginals_seed))
        if self.alpha is not None:
            parts.append(str(self.alpha))
        return "/".join(parts)
