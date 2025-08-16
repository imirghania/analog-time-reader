import argparse
from pathlib import Path
import shutil
from random import shuffle, sample
from ..config.config import PATHS


HOME = Path.cwd()
original_dataset_dir = HOME/PATHS.original_dataset_dir
annotations_dir = HOME/PATHS.yolo_annotations_dir
yolo_output_dir = PATHS.yolo_dataset_dir
yolo_output_dir_abs = HOME/yolo_output_dir


parser = argparse.ArgumentParser(
    prog="yolo-dataset-generator", 
    description="Generate a YOLO dataset by matching YOLO annotations to their corresponding image files.")

parser.add_argument("-a", "--annotations-dir",
        default=annotations_dir,
        help="The directory containing the YOLO annotations files.")

parser.add_argument("-r", "--raw-images-dir",
                default=original_dataset_dir,
                help="The directory containing the raw images.")

parser.add_argument("-o", "--output_dir",
                default=yolo_output_dir_abs,
                help="Path to the YOLO annotations output directory.")

parser.add_argument("-t", "--include_test",
                action='store_true',
                default=True,
                help="Indicates whether to include test images in the dataset.")

args = parser.parse_args()


def reset_yolo_dataset_structure(dataset_dir:Path,
                                include_test:bool=True):
    """
    Clears a YOLO dataset directory and resets it to the default empty structure.

    Args:
        dataset_dir (Path): Path to the root dataset directory.
    """
    # Define the expected structure
    structure = [
        dataset_dir / 'train' / 'images',
        dataset_dir / 'train' / 'labels',
        dataset_dir / 'val' / 'images',
        dataset_dir / 'val' / 'labels'
    ]
    
    if include_test:
        structure.append(dataset_dir / 'test' / 'images')
    
    # Remove the dataset directory if it exists
    if dataset_dir.exists() and dataset_dir.is_dir():
        shutil.rmtree(dataset_dir)

    # Recreate the directory structure
    for path in structure:
        path.mkdir(parents=True, exist_ok=True)

    print(f"YOLO dataset directory reset at: {dataset_dir}")


def generate_yolo_dataset(original_images_dir:Path, 
                        annotations_dir:Path, 
                        output_dir:Path,
                        train_images_number:int,
                        include_test:bool=True):
    if not original_images_dir.is_dir():
        raise FileNotFoundError(f"Raw images directory not found: {original_images_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"YOLO annotations directory not found: {annotations_dir}")
    
    reset_yolo_dataset_structure(output_dir, 
                                include_test=include_test)
    
    paths = list(annotations_dir.iterdir())
    shuffled_annotation_paths = paths.copy()
    shuffle(shuffled_annotation_paths)

    annotated_image_names = [p.stem for p in paths]
    unseen_image_names = [
        img_path for img_path in list(original_dataset_dir.iterdir())
        if img_path.stem not in annotated_image_names
        ]

    annotation_paths_train = shuffled_annotation_paths[:train_images_number]
    annotation_paths_val = shuffled_annotation_paths[train_images_number:]
    image_paths_test = sample(unseen_image_names, 100)

    annotation_groups_map = {
        "train": annotation_paths_train,
        "val": annotation_paths_val,
    }
    
    if include_test:
        annotation_groups_map.update({"test": image_paths_test})

    for group, paths in annotation_groups_map.items():
        image_group_path = HOME/f"{yolo_output_dir}/{group}/images"
        label_group_path = HOME/f"{yolo_output_dir}/{group}/labels"
        for path in paths:
            if group.lower() == "test":
                shutil.copy2(path, image_group_path)
            else:
                image_path = (
                    HOME/f"{original_dataset_dir}/{path.stem}.jpg"
                    )
                shutil.copy2(image_path, image_group_path)
                shutil.copy2(path, label_group_path)


if __name__ == "__main__":
    import yaml
    params = yaml.safe_load(
        open(HOME/"params.yaml")
        )["yolo_dataset_gen"]
    generate_yolo_dataset(Path(args.raw_images_dir),
                        Path(args.annotations_dir), 
                        Path(args.output_dir),
                        params["train_imgs_num"],
                        args.include_test)