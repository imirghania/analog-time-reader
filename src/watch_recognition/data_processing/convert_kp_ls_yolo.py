import argparse
import json
import os
from pathlib import Path
import shutil
from ..config.config import PATHS


HOME = Path.cwd()

parser = argparse.ArgumentParser(
    prog="annotation-converter", 
    description="Convert label-studio annotations to YOLO format.")

parser.add_argument("input",
                help="The name of the label-studio annotations file.")

# output_dir = HOME/"annotations/keypoints/yolo"
output_dir = HOME/PATHS.output_dir
parser.add_argument("-o", "--output_dir",
                default=output_dir,
                help="Path to the YOLO annotations output directory.")

parser.add_argument("-d", "--dimensions",
                type=int,
                choices=range(2,4),
                default=3,
                help="Indicates whether to include the visibilty for each key-point coordinates. 2 indicates no visibility (Only coordinates), 3 indicates visibility.")

args = parser.parse_args()

# ls_annotations_file = HOME/f"annotations/labelstudio/{args.input}"
ls_annotations_file = f"{PATHS.ls_annotations_dir}/{args.input}"
ls_annotations_file_abs = HOME / ls_annotations_file

def convert_ls_to_yolo_kpt(ls_json_file, output_dir, class_map, dims=3):
    # Ensure output directory exists and empty
    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError as e:
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

    if not ls_json_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ls_json_file}")

    try:
        with open(ls_json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {ls_annotations_file_abs}") from e

    for task in data:
        image_path = task['data']['image']
        image_name = os.path.basename(image_path)
        txt_filename = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        annotations = task.get('annotations', [])
        if not annotations:
            continue
        annotation = annotations[0]  # Assuming one annotation per task

        image_width = None
        image_height = None
        # Collect image dimensions
        for res in annotation['result']:
            if res['type'] == 'rectanglelabels':
                image_width = res['original_width']
                image_height = res['original_height']
                break
        if image_width is None or image_height is None:
            continue  # Cannot process without image dimensions

        objects = []

        # Map rectangle IDs to their data
        rectangles = {}
        for result in annotation['result']:
            if result['type'] == 'rectanglelabels':
                bbox = result
                bbox_id = bbox['id']
                bbox_label = bbox['value']['rectanglelabels'][0]
                class_idx = class_map.get(bbox_label)
                if class_idx is None:
                    continue  # Skip unknown classes

                x = bbox['value']['x'] / 100
                y = bbox['value']['y'] / 100
                width = bbox['value']['width'] / 100
                height = bbox['value']['height'] / 100
                # Convert from top-left corner to center coordinates
                x_center = x + width / 2
                y_center = y + height / 2

                rectangles[bbox_id] = {
                    'class_idx': class_idx,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'keypoints': []
                }

        # Collect keypoints associated with each rectangle
        for kp_result in annotation['result']:
            if kp_result['type'] == 'keypointlabels':
                parent_id = kp_result.get('parentID')
                if parent_id in rectangles:
                    kp_x = kp_result['value']['x'] / 100
                    kp_y = kp_result['value']['y'] / 100
                    # Optional: Assign visibility (0: not labeled, 1: labeled but not visible, 2: visible)
                    visibility = 2  # Assuming keypoints are visible
                    if dims == 3:
                        rectangles[parent_id]['keypoints'].extend([kp_x, kp_y, visibility])
                    else:
                        rectangles[parent_id]['keypoints'].extend([kp_x, kp_y])

        # Prepare YOLO formatted lines
        lines = []
        for rect in rectangles.values():
            obj = [
                rect['class_idx'],
                rect['x_center'],
                rect['y_center'],
                rect['width'],
                rect['height']
            ] + rect['keypoints']
            line = ' '.join(map(str, obj))
            lines.append(line)

        # Write to YOLO format file
        with open(txt_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')


class_map = {'clock': 0}

if __name__ == "__main__":
    convert_ls_to_yolo_kpt(ls_annotations_file_abs, 
                        args.output_dir,
                        class_map,
                        args.dimensions)
    annotations_num = len(list(Path(HOME/output_dir).glob("*.txt")))
    print(f"[OUTPUT DIR ANNOTATIONS NUM] {annotations_num}")