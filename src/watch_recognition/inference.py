import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO

from .geometry_elements import Line, Point
from .utils import points_to_time


def estimate_time(keypoints):
    """Estimate time from keypoints."""
    twelve, center, hour, minute, three, six, nine = keypoints
    
    twelve_oclock = Point(*twelve)
    clock_center = Point(*center)
    hour_tip = Point(*hour)
    minute_tip = Point(*minute)
    
    hour_read, minutes_read = points_to_time(
        top=twelve_oclock,
        center=clock_center, 
        hour=hour_tip, 
        minute=minute_tip
    )
    
    return (
        hour_read.item(), 
        minutes_read.item(),
        clock_center,
        hour_tip,
        minute_tip
        )


def draw_clock_hands(image:Image, 
                    clock_center:Point, 
                    hour_tip:Point, 
                    minute_tip:Point):
    """Draw hour and minute hands on the image."""
    hour_line = Line(clock_center, hour_tip)
    minute_line= Line(clock_center, minute_tip)
    
    annotated_image = hour_line.draw(np.array(image), (255, 0, 0), 4)
    annotated_image = minute_line.draw(annotated_image, (0, 0, 255), 2)
    
    return annotated_image


def annotate_image(input:Path, 
                output_dir:Path, 
                model:Path, 
                home:Path|None=None,
                draw_hands:bool=False):
    HOME = Path.cwd() if home is None else Path(home)
    model_path = HOME / model

    model = YOLO(model_path)
    input_image = Image.open(input)
    result = model.predict(input_image)[0]
    
    detections = sv.Detections.from_ultralytics(result)
    keypoints = sv.KeyPoints.from_ultralytics(result)
    
    # Estimate time for each detection and get hand positions
    time_readings = []
    hand_positions = []
    
    for kps in result.keypoints:
        (hour_read, 
        minutes_read, 
        clock_center, 
        hour_tip, 
        minute_tip) = estimate_time(kps.xy[0].tolist())
        
        time_readings.append((hour_read, minutes_read))
        hand_positions.append((clock_center, hour_tip, minute_tip))
    
    labels = [f"{tr[0]:02d}:{tr[1]:02d}" for tr in time_readings]
    
    image_np = np.array(input_image)
    
    # Annotate image
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image_np.copy(),
        detections=detections
    )
    
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.GREEN,
        radius=4
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=keypoints
    )
    
    if draw_hands:
        for clock_center, hour_tip, minute_tip in hand_positions:
            annotated_frame = draw_clock_hands(
                annotated_frame, clock_center, hour_tip, minute_tip
            )
    
    label_annotator = sv.LabelAnnotator()
    annotated_image = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections, 
        labels=labels
    )
    
    # Convert back to PIL Image and save
    # if annotated_image.shape[-1] == 3:
    #     annotated_image_rgb = cv2.cvtColor(
    #         annotated_image, cv2.COLOR_BGR2RGB)
    # else:
    #     annotated_image_rgb = annotated_image
    # output_image = Image.fromarray(annotated_image_rgb)
    
    output_image = Image.fromarray(annotated_image)
    output_path = output_dir / input.name
    output_image.save(output_path)
    
    print(f"Successfully processed image. Saved result to: {output_path}")
    print("Detected times:")
    for i, (hour, minute) in enumerate(time_readings):
        print(f"  Watch {i+1}: {hour:02d}:{minute:02d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect analog watches and read time from images"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Path to save output image with detection results"
    )
    parser.add_argument(
        "-m", "--model",
        default="models/keypoints/best.pt",
        help="Path to YOLO model (default: models/keypoints/best.pt)"
    )
    parser.add_argument(
        "--home",
        default=Path.cwd(),
        type=Path,
        help="Base directory for relative paths (default: project root)"
    )
    parser.add_argument(
        "--draw-hands",
        action="store_true",
        help="Draw lines for hour and minute hands"
    )
    
    args = parser.parse_args()
    
    annotate_image(Path(args.input), 
                Path(args.output), 
                Path(args.model)
                )