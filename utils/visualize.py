from os import PathLike
from typing import List, Literal, Optional, Union

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


def restore_rotation(tensor: Tensor, theta: Tensor) -> Tensor:
    assert tensor.shape[-1] == 2

    rotation_mat = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ],
        device=tensor.device,
    )
    return tensor @ rotation_mat.inverse()


def get_svg_node_rectangles(
    positions,
    nodes_weight,
    mask_valid,
    *,
    av_index,
    agent_index,
    width=4,
    height=2,
) -> List[str]:
    rect_tags = []
    color_dict = {agent_index: "red"}
    if nodes_weight is None:
        nodes_weight = torch.tensor([1.0 for _ in range(mask_valid.shape[0])])

    angles = torch.atan2(
        positions[:, 20, 1] - positions[:, 19, 1],
        positions[:, 20, 0] - positions[:, 19, 0],
    )

    rect_tags = [
        f"""<rect x="{x - width/2}" y="{y - height/2}"
                                width="{width}" height="{height}"
                                stroke="black" stroke-width="0.1"
                                fill="{color_dict.get(i, f'hsl(210, 100%, {100 - nodes_weight[i]*70}%)')}"
                                opacity="0.95"
                                transform="rotate({angle * 57.2958}, {x}, {y})"/>"""
        for i, ((x, y), angle, valid) in enumerate(
            zip(positions[:, 19, :], angles, mask_valid)
        )
        if valid
    ]
    return rect_tags


def get_svg_map_centerlines(
    positions,
    origin,
    city: str,
) -> List[str]:
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
                np.array(
                    [lane_cl[:, 0] - origin[0], lane_cl[:, 1] - origin[1]]
                )
            )

    centerline_tags = [
        f"""<polyline points="{','.join([f'{x},{y}' for x, y in zip(lane_cl[0], lane_cl[1])])}"
                style="fill:none;stroke:black;stroke-width:0.1"/>"""
        for lane_cl in lane_centerlines
    ]
    return centerline_tags


def get_svg_trajectories(
    trajectories,
    probs,
    positions,
    mask_valid,
    *,
    agent_index,
    av_index,
    width=4,
    height=2,
    style: Union[Literal["points"], Literal["lines"]] = "lines",
) -> List[str]:
    trajectory_tags: List[str] = []
    gt_tags: List[str] = []
    for node_index, (node_traj, node_positions) in enumerate(
        zip(trajectories, positions)
    ):
        if mask_valid[node_index]:
            node_origin = node_positions[19]
            if style == "points":
                for mode_i, mode in enumerate(node_traj):
                    probability: float = probs[node_index, mode_i].item()
                    for x, y in mode:
                        trajectory_tags.append(
                            f"""<circle cx="{x + node_origin[0]}" cy="{y + node_origin[1]}" r="0.3" fill="hsl(210, 100%, {100 - probability*100 * 2}%)" />"""
                        )
                for timestep, (x, y) in enumerate(node_positions):
                    gt_tags.append(
                        f"""<circle cx="{x}" cy="{y}" r="0.3" fill="{'hsl(0, 100%, 50%)' if timestep > 19 else 'hsl(180,0%,50%)' }" />"""
                    )
            if style == "lines":
                tags = [
                    (
                        f"""<polyline points="{','.join([f'{x + node_origin[0]},{y + node_origin[1]}' for x, y in mode])}"
                            style="fill:none;stroke:hsl(210, 100%, {110 - probs[node_index, mode_i].item()*100 * 2.5}%);
                            opacity:{0.2+probs[node_index, mode_i].item()*3};stroke-width:0.8"
                            stroke-linecap="round"/>""",
                        probs[node_index, mode_i],
                    )
                    for mode_i, mode in enumerate(node_traj)
                ]
                tags.sort(key=lambda x: x[1])
                trajectory_tags.extend(map(lambda t: t[0], tags))

                gt_tags.extend(
                    [
                        f"""<polyline points="{','.join([f'{x},{y}' for x, y in node_positions[0:20]])}"
                        style="fill:none;stroke:hsl(180, 0%, 50%);stroke-width:0.3"/>""",
                        f"""<polyline points="{','.join([f'{x},{y}' for x, y in node_positions[20:]])}"
                        style="fill:none;stroke:hsl(0, 100%, 50%);stroke-width:0.3"/>""",
                    ]
                )
    trajectory_tags.extend(gt_tags)
    return trajectory_tags
