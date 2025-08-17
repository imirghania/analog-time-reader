import dataclasses
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy import odr
from skimage.measure import (
    LineModelND,
    ransac,
)


@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float
    name: str = ""  # TODO name and score could be moved to a separate class
    score: Optional[float] = None

    @classmethod
    def none(cls) -> "Point":
        return Point(0.0, 0.0, "", 0.0)

    def scale(self, x: float, y: float) -> "Point":
        return Point(self.x * x, self.y * y, self.name, self.score)

    def translate(self, x: float, y: float) -> "Point":
        return Point(self.x + x, self.y + y, self.name, self.score)

    def distance(self, other: "Point") -> float:
        diff = np.array(self.as_coordinates_tuple) - np.array(
            other.as_coordinates_tuple
        )
        return float(np.sqrt((diff**2).sum()))

    def rotate_around_point(self, origin: "Point", angle: float) -> "Point":
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(
            [
                [c, -s],
                [s, c],
            ]
        )
        point = np.array([self.as_coordinates_tuple]).T
        origin = np.array([origin.as_coordinates_tuple]).T
        rotated = (R @ (point - origin) + origin).flatten()
        return Point(rotated[0], rotated[1], self.name, self.score)

    @property
    def as_coordinates_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    @property
    def as_array(self) -> np.ndarray:
        return np.array(self.as_coordinates_tuple)

    def plot(self, ax=None, color: Optional[str] = None, marker="x", size=20, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(
            self.x,
            self.y,
            label=self.name,
            color=color,
            marker=marker,
            s=size,
            **kwargs,
        )

    def draw_marker(
        self,
        image: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: int = 3,
    ) -> np.ndarray:
        original_image_np = image.astype(np.uint8)

        x, y = self.as_coordinates_tuple
        x = int(x)
        y = int(y)

        cv2.drawMarker(
            original_image_np, (x, y), color, cv2.MARKER_CROSS, thickness=thickness
        )

        return original_image_np.astype(np.uint8)

    def draw(self, image: np.ndarray, color=None) -> np.ndarray:
        if color is not None:
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"To draw colored point on image, it has to have 3 channels. Got image with shape {image.shape}"
                )
            value = color
        else:
            value = 1
        image = image.copy()
        image[self.y, self.x] = value
        return image

    def rename(self, new_name: str) -> "Point":
        return dataclasses.replace(self, name=new_name)

    @property
    def center(self) -> "Point":
        return self


@dataclasses.dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: str = ""
    score: Optional[float] = None

    def __post_init__(self):
        x_min, x_max = min(self.x_min, self.x_max), max(self.x_min, self.x_max)
        y_min, y_max = min(self.y_min, self.y_max), max(self.y_min, self.y_max)
        object.__setattr__(self, "x_min", x_min)
        object.__setattr__(self, "x_max", x_max)
        object.__setattr__(self, "y_min", y_min)
        object.__setattr__(self, "y_max", y_max)

    @classmethod
    def unit(cls, name="", score=None):
        return BBox(
            x_min=0,
            y_min=0,
            x_max=1,
            y_max=1,
            name=name,
            score=score,
        )

    @classmethod
    def from_center_width_height(
        cls, center_x, center_y, width, height, name="", score=None
    ):
        return BBox(
            x_min=center_x - width / 2,
            y_min=center_y - height / 2,
            x_max=center_x + width / 2,
            y_max=center_y + height / 2,
            name=name,
            score=score,
        )

    @classmethod
    def from_ltwh(cls, left, top, width, height, name="", score=None):
        return BBox(
            x_min=left,
            y_min=top,
            x_max=left + width,
            y_max=top + height,
            name=name,
            score=score,
        )

    def contains(self, point: Point) -> bool:
        contains_x = self.x_min < point.x < self.x_max
        contains_y = self.y_min < point.y < self.y_max
        return contains_x and contains_y

    def scale(self, x: float, y: float) -> "BBox":
        return BBox(
            self.x_min * x,
            self.y_min * y,
            self.x_max * x,
            self.y_max * y,
            self.name,
            self.score,
        )

    @property
    def center(self) -> "Point":
        return Point((self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2)

    def center_scale(self, x: float, y: float) -> "BBox":
        w, h = self.width * x, self.height * y
        cx, cy = self.center.x, self.center.y

        x_min = cx - w / 2
        x_max = cx + w / 2

        y_min = cy - h / 2
        y_max = cy + h / 2

        return dataclasses.replace(
            self, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )

    def translate(self, x: float = 0.0, y: float = 0.0) -> "BBox":
        return BBox(
            x_min=self.x_min + x,
            x_max=self.x_max + x,
            y_min=self.y_min + y,
            y_max=self.y_max + y,
            score=self.score,
            name=self.name,
        )

    def reflect(self, x: Optional[float] = None, y: Optional[float] = None) -> "BBox":
        # TODO this should be more generic
        if x is not None:
            reflected_box = BBox(
                x_min=2 * x - self.x_min,
                x_max=2 * x - self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                score=self.score,
                name=self.name,
            )
        if y is not None:
            reflected_box = BBox(
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=2 * y - self.y_min,
                y_max=2 * y - self.y_max,
                score=self.score,
                name=self.name,
            )
        return reflected_box

    @property
    def as_coordinates_tuple(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def convert_to_int_coordinates_tuple(
        self, method: str = "round"
    ) -> Tuple[int, int, int, int]:
        if method == "round":
            return tuple(np.round(self.as_coordinates_tuple).astype(int))
        elif method == "floor":
            return tuple(np.floor(self.as_coordinates_tuple).astype(int))
        else:
            raise ValueError(f"unrecognized method {method}, choose one of round|floor")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self):
        return self.height * self.width

    @property
    def top(self):
        return self.y_min

    @property
    def bottom(self):
        return self.y_max

    @property
    def left(self):
        return self.x_min

    @property
    def right(self):
        return self.x_max

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def corners(self) -> List[Point]:
        return [
            Point(self.x_min, self.y_min),
            Point(self.x_max, self.y_min),
            Point(self.x_max, self.y_max),
            Point(self.y_min, self.y_max),
        ]


    def intersection(self, other: "BBox") -> "BBox":
        x_min = max(self.x_min, other.x_min)
        x_max = min(self.x_max, other.x_max)

        y_min = max(self.y_min, other.y_min)
        y_max = min(self.y_max, other.y_max)

        width = x_max - x_min
        height = y_max - y_min
        if width < 0 or height < 0:
            x_min = 0
            y_min = 0
            x_max = 0
            y_max = 0
        return BBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            name=self.name,
            score=self.score,
        )

    def iou(self, other: "BBox") -> float:
        intersection_area = self.intersection(other).area
        if not intersection_area:
            return 0.0
        union_area = self.area + other.area - intersection_area
        return intersection_area / union_area

    def plot(
        self,
        ax=None,
        color: str = "red",
        linewidth: int = 1,
        draw_label: bool = True,
        draw_score: bool = False,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()
        rect = patches.Rectangle(
            (self.left, self.top),
            self.width,
            self.height,
            edgecolor=color,
            facecolor="none",
            linewidth=linewidth,
            **kwargs,
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        text = ""
        if self.name:
            text += self.name
        if draw_score:
            text += f"\n{self.score:.2f}"

        border_offset = 50
        y_text_pos = (
            self.y_min - border_offset
            if self.y_min - border_offset > border_offset
            else self.y_min + border_offset
        )
        if draw_label or draw_score:
            ax.text(
                self.x_min,
                y_text_pos,
                text,
                bbox={"facecolor": color, "alpha": 0.4},
                clip_box=ax.clipbox,
                clip_on=True,
            )

    def draw(
        self, image: np.ndarray, color: Tuple[int, int, int] = (255, 0, 255)
    ) -> np.ndarray:
        original_image_np = image.astype(np.uint8)
        xmin, ymin, xmax, ymax = tuple(map(int, self.as_coordinates_tuple))

        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        border_offset = 50
        y_text_pos = (
            ymin - border_offset
            if ymin - border_offset > border_offset
            else ymin + border_offset
        )
        cv2.putText(
            original_image_np,
            self.name,
            (xmin, y_text_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        return original_image_np.astype(np.uint8)

    def rename(self, new_name: str) -> "BBox":
        return dataclasses.replace(self, name=new_name)


@dataclasses.dataclass(frozen=True)
class Line:
    start: Point
    end: Point
    score: float = 0

    @classmethod
    def from_multiple_points(
        cls, points: List[Point], use_ransac: bool = False
    ) -> "Line":
        if len(points) < 2:
            raise ValueError(f"Need at least 2 points to fit a lint, got {len(points)}")
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        # vertical line
        # TODO < 2 is still pretty broad, maybe it should be less than 1e-3?
        if np.std(x_coords) < 2:
            x_const = float(np.mean(x_coords))
            return Line(Point(x_const, y_min), Point(x_const, y_max))
        # horizontal line
        if np.std(y_coords) < 2:
            y_const = float(np.mean(y_coords))
            return Line(Point(x_min, y_const), Point(x_max, y_const))

        if use_ransac and len(points) > 2:
            data = np.column_stack([x_coords, y_coords])
            model_robust, inliers = ransac(
                data, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000
            )
            line_x = [x_min, x_max]
            line_y_min, line_y_max = model_robust.predict_y(line_x)
            start = Point(x_min, line_y_min)
            end = Point(x_max, line_y_max)
        else:
            # other cases
            poly1d = cls._fit_line(x_coords, y_coords)
            start = Point(x_min, poly1d(x_min))
            end = Point(x_max, poly1d(x_max))
        window = BBox(x_min, y_min, x_max, y_max, "")
        return Line(start, end).clip(window)

    @property
    def poly1d(self) -> np.poly1d:
        x_coords = [self.start.x, self.end.x]
        y_coords = [self.start.y, self.end.y]
        return self._fit_line(x_coords, y_coords)

    @classmethod
    def _fit_line(
        cls, x_coords: Union[List, np.ndarray], y_coords: Union[List, np.ndarray]
    ) -> np.poly1d:
        """
        Fit 1st degree polynomial using ODR = Orthogonal Distance Regression
        Least squares regression won't work for perfectly vertical lines.

        Notes:
          Check out this Stack Overflow question and answer:
            https://stackoverflow.com/a/10982488/8814045
          and scipy docs with an example:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.polynomial.html#scipy.odr.polynomial

        Args:
            x_coords:
            y_coords:

        Returns:

        """
        # poly_coeffs = np.polyfit(x_coords, y_coords, deg=1)
        # return np.poly1d(poly_coeffs)

        poly_model = odr.polynomial(1)
        data = odr.Data(x_coords, y_coords)
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()
        poly = np.poly1d(output.beta[::-1])

        return poly

    @property
    def slope(self) -> float:
        return self.poly1d.coeffs[0]

    @property
    def vector(self) -> np.ndarray:
        return self.end.as_array - self.start.as_array

    @property
    def unit_vector(self) -> np.ndarray:
        vector = self.vector
        return vector / np.linalg.norm(vector)

    @property
    def center(self) -> Point:
        x = (self.start.x + self.end.x) / 2
        y = (self.start.y + self.end.y) / 2
        return Point(x=x, y=y)

    @property
    def angle(self):
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return np.arctan2(dy, dx)

    def angle_between(self, other):
        vec_1 = self.vector
        vec_2 = other.vector
        unit_vector_1 = vec_1 / np.linalg.norm(vec_1)
        unit_vector_2 = vec_2 / np.linalg.norm(vec_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        return angle

    @property
    def length(self) -> float:
        return self.start.distance(self.end)

    def scale(self, x: float, y: float) -> "Line":
        return Line(self.start.scale(x, y), self.end.scale(x, y), score=self.score)

    def translate(self, x: float, y: float) -> "Line":
        return Line(
            self.start.translate(x, y), self.end.translate(x, y), score=self.score
        )

    def projection_point(self, point: Point) -> Point:
        line_fit = self.poly1d
        m = line_fit.coeffs[0]
        k = line_fit.coeffs[1]
        proj_point_x = (point.x + m * point.y - m * k) / (m**2 + 1)
        proj_point_y = m * proj_point_x + k
        return Point(proj_point_x, proj_point_y)

    def plot(self, ax=None, color=None, draw_arrow: bool = True, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(
            [self.start.x, self.end.x],
            [self.start.y, self.end.y],
            color=color,
            **kwargs,
        )

        if draw_arrow:
            dx = np.sign(self.unit_vector[0])
            dy = self.slope * dx
            ax.arrow(
                self.center.x,
                self.center.y,
                dx,
                dy,
                shape="full",
                edgecolor="black",
                facecolor=color,
                width=0.5,
            )

    def draw(
        self, img: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness=10
    ) -> np.ndarray:
        original_image_np = img.astype(np.uint8)
        start = self.start.as_coordinates_tuple
        end = self.end.as_coordinates_tuple
        start = tuple(map(int, start))
        end = tuple(map(int, end))
        cv2.line(original_image_np, start, end, color, thickness=thickness)
        return original_image_np.astype(np.uint8)

    def clip(self, bbox: BBox) -> "Line":
        start, end = self.start, self.end
        new_points = []
        for p in (start, end):
            x = min(max(p.x, bbox.x_min), bbox.x_max)
            y = min(max(p.y, bbox.y_min), bbox.y_max)
            new_p = Point(x, y)
            new_points.append(new_p)
        start, end = new_points
        return Line(start, end, score=self.score)