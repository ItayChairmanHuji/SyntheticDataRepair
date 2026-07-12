import time

import hydra
import pandas as pd
from omegaconf import DictConfig

from denial_constraints.denial_constraints import DenialConstraints
from graph.graph import Graph
from graph.initializer import GraphBuilder
from violations_detection.violations_finder import find_violations


def find_cover(graph: Graph):
    cover = []
    while graph.has_edges():
        selected = graph.pick_random_edge()
        graph.remove_vertex(selected[0])
        cover.append(selected[0])
        graph.remove_vertex(selected[1])
        cover.append(selected[1])
    return cover


def repair(data: pd.DataFrame, dcs: DenialConstraints):
    violations_set = find_violations(data, dcs)
    graph = GraphBuilder(len(data), violations_set).graph
    cover = find_cover(graph)
    return data.drop(index=cover)


def classic_vc_repair(data: pd.DataFrame, dcs: DenialConstraints):
    tick = time.perf_counter()
    repaired_data = repair(data, dcs)
    tock = time.perf_counter()
    return repaired_data, tock - tick


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    synth_data = pd.read_csv(cfg.input_path)
    dcs = DenialConstraints.from_file(cfg.dcs_file)
    output = classic_vc_repair(synth_data, dcs)
    output[0].to_csv(cfg.output_path, index=False)
    print(f"Repair time: {output[1]:.2f} seconds")
    return output

if __name__ == "__main__":
    main(DictConfig({
        "input_path": "resources/data/data.csv",
        "dcs_file": "resources/constraints/adult.txt",
        "output_path": "resources/data/repaired.csv"
    }))
