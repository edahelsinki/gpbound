from pathlib import Path
import json


def load_config(path_to_config: Path):
    path_to_config = Path(path_to_config).resolve()
    with open(path_to_config) as f:
        config = json.load(f)
    for k in ["dir_data", "dir_results", "dir_figures", "dir_tables"]:
        config[k] = path_to_config.parent / config[k]
    return config
