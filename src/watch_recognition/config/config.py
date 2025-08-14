from pathlib import Path
import yaml
from typing import Any
from dataclasses import dataclass, asdict


def load_config(config_file) -> dict[str, Any]:
    """Load and parse the YAML configuration file."""
    config_path = Path(__file__).parent / config_file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")
    
    return config


config = load_config("config.yaml")


@dataclass(frozen=True)
class ProjectPaths:
    """Stores all project paths with type safety."""
    original_dataset_dir: Path
    yolo_dataset_dir: Path
    yolo_annotations_dir: Path
    ls_annotations_dir: Path
    
    @classmethod
    def from_config(cls, config: dict) -> 'ProjectPaths':
        """Alternative constructor from config dict."""
        return cls(
            original_dataset_dir=(
                Path(config["paths"]["original_dataset_dir"])),
            yolo_dataset_dir=(
                Path(config["paths"]["yolo_dataset_dir"])),
            yolo_annotations_dir=(
                Path(config["paths"]["yolo_annotations_dir"])),
            ls_annotations_dir=Path(config["paths"]["ls_annotations_dir"])
        )

PATHS = ProjectPaths.from_config(config)

def validate_paths() -> None:
    """Verify that all configured paths exist."""
    for name, path in asdict(PATHS).items():
        if not path.exists():
            raise FileNotFoundError(f"Configured path '{name}' does not exist: {path}")
    print("All setup directories do exist.")


if __name__ == "__main__":
    validate_paths()