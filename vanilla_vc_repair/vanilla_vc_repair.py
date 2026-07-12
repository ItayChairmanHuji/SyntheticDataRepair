import time

import hydra
import pandas as pd
from omegaconf import DictConfig

from denial_constraints.denial_constraints import DenialConstraints
from graph.builder import ConflictGraphBuilder
from graph.graph import Graph
from violations_detection.violations_finder import find_violations


def find_cover(graph: Graph):
    cover = []
    while graph.has_edges():
        selected = int(graph.degrees.argmax())
        cover.append(selected)
        graph.remove_vertex(selected)
    return cover


def repair(data: pd.DataFrame, dcs: DenialConstraints) -> pd.DataFrame:
    violations_set = find_violations(data, dcs)
    graph = ConflictGraphBuilder.build(len(data), violations_set)
    cover = find_cover(graph)
    return data.drop(index=cover)


def vanilla_vc_repair(data: pd.DataFrame, dcs: DenialConstraints):
    tick = time.perf_counter()
    repaired_data = repair(data, dcs)
    tock = time.perf_counter()
    return repaired_data, tock - tick


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    synth_data = pd.read_csv(cfg.input_path)
    dcs = DenialConstraints(cfg.dcs_file)
    output = vanilla_vc_repair(synth_data, dcs)
    output[0].to_csv(cfg.output_path, index=False)
    print(f"Repair time: {output[1]:.2f} seconds")
    return output

if __name__ == "__main__":
    main(DictConfig({
        "input_path": "resources/data/private/adult.csv",
        "output_path": "resources/models/repaired.csv",
        "dcs_file": "resources/dcs/adult.dcs",
    }))
