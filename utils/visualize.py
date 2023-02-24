from os import PathLike
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from argoverse.map_representation.map_api import ArgoverseMap

AVM = ArgoverseMap()


class SeqTrajectory:
    seq_id: int
    traj_tensor: Tensor
    prob_tensor: Tensor

    def __init__(self, seq_id: int, traj_tensor: Tensor, prob_tensor: Tensor):
        self.seq_id = seq_id
        self.traj_tensor = traj_tensor
        self.prob_tensor = prob_tensor


class VisualizeOption:
    """
    Args:
        trajectory(bool): Whether to visualize the trajectory.
        node(bool): Whether to visualize the node.
        centerline(bool): Whether to visualize the centerline.
        svg_filepath(PathLike): The path to the svg file.
    """

    trajectory: bool
    node: bool
    centerline: bool
    width: float
    height: float
    svg_filepath: Optional[PathLike]

    def __init__(
        self,
        trajectory: bool = True,
        node: bool = True,
        centerline: bool = True,
        svg_filepath: Optional[PathLike] = None,
    ):
        self.trajectory = trajectory
        self.node = node
        self.centerline = centerline
        self.svg_filepath = svg_filepath


def restore_rotation(tensor: Tensor, theta: Tensor) -> Tensor:
    assert tensor.shape[-1] == 2

    rotation_mat = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ]
    )
    return tensor @ rotation_mat.inverse()


def get_svg_node_rectangles(
    positions, angles, nodes_weight, *, av_index, agent_index, width=4, height=2
):
    rect_tags = []
    color_dict = {agent_index: "red", av_index: "blue"}
    for i, ((x, y), angle) in enumerate(zip(positions[:, 0, :], angles)):
        rect_tag = f"""<rect x="{x - width/2}" y="{y - height/2}"
                                width="{width}" height="{height}"
                                stroke="black" stroke-width="{0.1 + nodes_weight[i] * 0.5}"
                                fill="{color_dict.get(i, 'transparent')}"
                                transform="rotate({-angle * 57.2958}, {x}, {y})"/>"""
        rect_tags.append(rect_tag)
    return rect_tags


def get_svg_map_centerlines(
    positions,
    origin,
    city: str,
) -> list:
    seq_lane_props = AVM.city_lane_centerlines_dict[city]

    x_max = positions[:, :, 0].max()
    x_min = positions[:, :, 0].min()
    y_max = positions[:, :, 1].max()
    y_min = positions[:, :, 1].min()

    lane_centerlines = []
    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline

        if (
            np.min(lane_cl[:, 0]) - origin[0] < x_max
            and np.min(lane_cl[:, 1]) - origin[1] < y_max
            and np.max(lane_cl[:, 0]) - origin[0] > x_min
            and np.max(lane_cl[:, 1]) - origin[1] > y_min
        ):
            lane_centerlines.append(
                np.array([lane_cl[:, 0] - origin[0], lane_cl[:, 1] - origin[1]])
            )

    centerline_tags = [
        f"""<polyline points="{','.join([f'{x},{y}' for x, y in zip(lane_cl[0], lane_cl[1])])}"
                style="fill:none;stroke:black;stroke-width:0.1"/>"""
        for lane_cl in lane_centerlines
    ]
    return centerline_tags


def get_svg_trajectories(
    trajectories, probs, positions, *, agent_index, av_index, width=4, height=2
):
    trajectory_tags = []

    for node_index, (node_traj, node_positions) in enumerate(
        zip(trajectories, positions)
    ):
        node_origin = node_positions[0]
        # if node_index == agent_index or node_index == av_index:
        for mode_i, mode in enumerate(node_traj):
            probability: float = probs[node_index, mode_i].item()
            for x, y in mode:
                trajectory_tags.append(
                    f"""<circle cx="{x + node_origin[0]}" cy="{y + node_origin[1]}" r="0.3" fill="hsl(210, 100%, {100 - probability*100 * 2}%)" />"""
                )
    return trajectory_tags
