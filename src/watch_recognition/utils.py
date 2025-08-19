import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from .geometry_elements import Point, Line


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    https://stackoverflow.com/a/13849249
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def points_to_time(
    center: Point, hour: Point, minute: Point, top: Point, debug: bool = False
) -> Tuple[float, float]:
    if debug:
        fig, ax = plt.subplots(1, 1)
        top.plot(color="green")
        hour.plot(color="red")
        center.plot(color="k")
        minute.plot(color="orange")
        ax.invert_yaxis()
        ax.legend()

    hour = hour.translate(-center.x, -center.y)
    minute = minute.translate(-center.x, -center.y)
    top = top.translate(-center.x, -center.y)
    center = center.translate(-center.x, -center.y)
    up = center.translate(0, -10)
    line_up = Line(center, up)
    line_top = Line(center, top)
    if debug:
        fig, ax = plt.subplots(1, 1)
        top.plot(color="green")
        hour.plot(color="red")
        center.plot(color="k")
        minute.plot(color="orange")
        line_up.plot(draw_arrow=False, color="black")
        line_top.plot(draw_arrow=False, color="green")
        ax.invert_yaxis()
        ax.legend()
    rot_angle = -np.rad2deg(line_up.angle_between(line_top))
    if top.x < 0:
        rot_angle = -rot_angle
    top = top.rotate_around_point(center, rot_angle)
    minute = minute.rotate_around_point(center, rot_angle)
    hour = hour.rotate_around_point(center, rot_angle)

    if debug:
        fig, ax = plt.subplots(1, 1)
        top.plot(color="green")
        hour.plot(color="red")
        center.plot(color="k")
        minute.plot(color="orange")
        line_up = Line(center, up)
        line_top = Line(center, top)
        line_up.plot(draw_arrow=False, color="black")
        line_top.plot(draw_arrow=False, color="green")

        ax.invert_yaxis()
        ax.legend()
    hour = hour.as_array
    minute = minute.as_array
    top = top.as_array

    minute_deg = np.rad2deg(angle_between(top, minute))
    # TODO verify how to handle negative angles
    if minute[0] < top[0]:
        minute_deg = 360 - minute_deg
    read_minute = minute_deg / 360 * 60
    read_minute = np.floor(read_minute).astype(int)
    read_minute = read_minute % 60

    hour_deg = np.rad2deg(angle_between(top, hour))
    # TODO verify how to handle negative angles
    if hour[0] < top[0]:
        hour_deg = 360 - hour_deg
    # In case where the minute hand is close to 12
    # we can assume that the hour hand will be close to the next hour
    # to prevent incorrect hour reading we can move it back by 10 deg
    if read_minute > 45:
        hour_deg -= 10

    read_hour = hour_deg / 360 * 12
    read_hour = np.floor(read_hour).astype(int)
    read_hour = read_hour % 12
    if read_hour == 0:
        read_hour = 12
    return read_hour, read_minute