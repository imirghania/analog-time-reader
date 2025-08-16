import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from .config.config import PATHS
from dvclive import Live
from ultralytics import YOLO


HOME = Path.cwd()
yaml_file_path = HOME/f"{PATHS.config_file.parent}/data.yaml"


def clean_metrics_keys(metrics_dict):
    """Transform dictionary keys from 'metrics/precision(B)' to 'precision'."""
    cleaned = {}
    for key, value in metrics_dict.items():
        stripped = key.replace('metrics/', '')
        cleaned_key = re.sub(r'\([A-Z]\)$', '', stripped)
        cleaned[cleaned_key] = value
    return cleaned


def main(model_path: Path, data_yaml: Path = None):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, plots=True)
    
    metrics_dict = metrics.results_dict
    box_metrics = {k:v for k,v in metrics_dict.items() if "(B)" in k}
    pose_metrics = {k:v for k,v in metrics_dict.items() if "(P)" in k}
    
    clean_box_metrics = clean_metrics_keys(box_metrics)
    clean_pose_metrics = clean_metrics_keys(pose_metrics)
    
    cm_df = metrics.confusion_matrix.to_df()
    cm_values = cm_df[["clock", "background"]].values
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_values,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=["clock", "background"],
        yticklabels=["clock", "background"],
        linewidths=0.5,
        linecolor="grey"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    with Live() as live:
        live.summary = {
            "model_name": model_path.stem,
            "box": clean_box_metrics,
            "pose": clean_pose_metrics
        }
        live.log_image("confusion_matrix.png", plt.gcf())
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO Model Evaluation')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to YOLO model weights (e.g., models/keypoints/<model-name>.pt)'
    )
    parser.add_argument(
        '-d',
        '--data-yaml',
        type=str,
        default=yaml_file_path,
        help='Path to data.yaml config (default: looks for data.yaml in the config dir)'
    )
    
    args = parser.parse_args()
    model_path = Path(args.model_path)
    main(model_path, args.data_yaml)