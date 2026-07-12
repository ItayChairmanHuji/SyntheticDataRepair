import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dataset.metadata import Metadata
from dataset.resutls import Results
from denial_constraints import DenialConstraints
from marginals.marginals import Marginals


class Dataset:
    BASE_PATH = Path("resources/data/")
    BASE_MARGINALS_PATH = Path("resources/marginals/")

    def __init__(self, metadata: Metadata, base: str):
        self.metadata = metadata
        self.id = metadata.id
        self.base = base
        self.data = pd.read_csv(self.BASE_PATH / metadata.id / "data.csv")
        self.dcs = DenialConstraints.from_dataset_name(self.metadata.dataset_name)
        self.marginals = self._load_marginals()
        self._compact = {}

    def path(self):
        return self.BASE_PATH / self.base / self.id

    def load_private_data(self):
        metadata = Metadata(
            data_type="private",
            generation_model="",
            dataset_name=self.metadata.dataset_name,
            epsilon=-1,
            size=-1,
            synthetic_seed=-1,
        )
        return Dataset(metadata)

    def load_synthetic_data(self):
        metadata = replace(
            self.metadata,
            data_type="synthetic",
            num_of_marginals=None,
            marginals_seed=None,
            alpha=None,
        )
        return Dataset(metadata)

    def add_marginals(self, num_of_marginals: int, marginals_seed: int) -> None:
        self.metadata = replace(self.metadata, num_of_marginals=num_of_marginals, marginals_seed=marginals_seed)
        self.marginals = self._load_marginals()

    def compact(self, attributes: list[str] | None = None):
        from violations_detection.models import compact_data

        attrs = sorted(attributes) if attributes else sorted(self.dcs.attrs)
        key = tuple(attrs)
        if key not in self._compact:
            self._compact[key] = compact_data(self.data, attrs)
        return self._compact[key]

    def __len__(self) -> int:
        return len(self.data)

    def _load_marginals(self) -> Marginals | None:
        if self.metadata.num_of_marginals is None or self.metadata.marginals_seed is None:
            return None
        return Marginals(
            self.metadata.dataset_name,
            self.metadata.num_of_marginals,
            self.metadata.marginals_seed,
        )

    def add_results(self, results: Results) -> None:
        results_dir = self.BASE_PATH / self.id / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(
            results_dir / f"results_{self.metadata.num_of_marginals}_{self.metadata.marginals_seed}.json", "w"
        ) as f:
            json.dump(results.__dict__, f)

    @property
    def marginals_error(self) -> float:
        if self.marginals is None:
            return -1
        return self.marginals.error(self.data)
