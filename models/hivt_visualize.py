from dataclasses import dataclass
import itertools
from typing import List, Optional

import torch
from torch import Tensor

from models.hivt import HiVT
from utils import TemporalData


@dataclass
class Point:
    timestep: int
    x: float
    y: float


@dataclass
class TrajectoryMode:
    probability: float
    points: List[Point]

    @property
    def pi(self) -> float:
        return self.probability


@dataclass
class NodeTrajectory:
    is_av: bool
    modes: List[TrajectoryMode]


@dataclass
class SeqTrajectorySet:
    seq_id: int
    nodes_traj: List[NodeTrajectory]
    tensor: Tensor

    @property
    def av_traj(self) -> NodeTrajectory:
        return self.nodes_traj[0]

    @property
    def non_av_traj(self) -> List[NodeTrajectory]:
        return self.nodes_traj[1:]


def _pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class HiVTVisualize(HiVT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_step(  # type: ignore[override]
        self,
        data: TemporalData,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> List[SeqTrajectorySet]:
        """
        Returns:
            Dict[seq_ids, predicted values]
        """
        # Forwarding the data through the model
        y_hat: Tensor
        pi: Tensor
        y_hat, pi = self(data)

        # Info about the output data shape
        assert y_hat.shape == (
            self.num_modes,
            data.num_nodes,
            self.future_steps,
            4,
        )
        assert pi.shape == (data.num_nodes, self.num_modes)

        prob_i: Tensor = pi.softmax(dim=1)

        assert data.seq_id.ndim == 1
        assert data.seq_id.shape[0] == len(data.av_index)

        # For visualize, includes all node trajectories.
        nodes_traj: Tensor = y_hat[:, :, :, :2].transpose(0, 1)
        assert nodes_traj.shape == (
            data.num_nodes,
            self.num_modes,
            self.future_steps,
            2,
        )

        split_indices = [b - a for a, b in _pairwise(data.av_index)] + [
            len(nodes_traj) - data.av_index[-1]
        ]
        traj_sliced_per_seq = torch.split(nodes_traj, split_indices)
        prob_sliced_per_seq = torch.split(prob_i, split_indices)

        output: List[SeqTrajectorySet] = [
            SeqTrajectorySet(
                seq_id=seq_id,
                nodes_traj=[
                    NodeTrajectory(
                        is_av=True if i == 0 else False,
                        modes=[
                            TrajectoryMode(
                                probability=mode_prob,
                                points=[
                                    Point(
                                        timestep=timestep,
                                        x=x.item(),
                                        y=y.item(),
                                    )
                                    for timestep, (x, y) in enumerate(
                                        mode_points
                                    )
                                ],
                            )
                            for mode_points, mode_prob in zip(
                                node_trajs, node_probs
                            )
                        ],
                    )
                    for i, (node_trajs, node_probs) in enumerate(
                        zip(seq_trajs, seq_probs)
                    )
                ],
                tensor=seq_trajs,
            )
            for seq_id, seq_trajs, seq_probs in zip(
                data.seq_id, traj_sliced_per_seq, prob_sliced_per_seq
            )
        ]

        return output
