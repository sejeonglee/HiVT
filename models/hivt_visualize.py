from dataclasses import dataclass
import itertools
from typing import List, Optional

import torch
from torch import Tensor

from models.hivt import HiVT
from utils import TemporalData, visualize_seq_trajectory
from utils.visualize import SeqTrajectory


def _pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class HiVTVisualize(HiVT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        l2_norm = (
            torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask
        ).sum(
            dim=-1
        )  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log(
            "val_reg_loss",
            reg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        y_hat_agent = y_hat[:, data["agent_index"], :, :2]
        y_agent = data.y[data["agent_index"]]
        fde_agent = torch.norm(
            y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1
        )
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[
            best_mode_agent, torch.arange(data.num_graphs)
        ]

        # Visualize Codes!
        seq_trajectories = self.get_seqtrajectory_list(data, y_hat, pi)
        for seq in seq_trajectories:
            visualize_seq_trajectory(
                data,
                seq,
                weights=None,
                save_svg_path=None,
                traj_style="lines",
            )


    def get_seqtrajectory_list(self, data, y_hat, pi):
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

        # Restore per-node rotation by self.rotate
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data["rotate_angles"])
            cos_vals = torch.cos(data["rotate_angles"])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals

            nodes_traj = torch.bmm(
                nodes_traj.reshape(data.num_nodes, -1, 2),
                rotate_mat.inverse(),
            ).reshape(nodes_traj.shape)

        split_indices = [b - a for a, b in _pairwise(data.av_index)] + [
            len(nodes_traj) - data.av_index[-1]
        ]
        traj_sliced_per_seq = torch.split(nodes_traj, split_indices)
        prob_sliced_per_seq = torch.split(prob_i, split_indices)

        output: List[SeqTrajectory] = [
            SeqTrajectory(
                seq_id=seq_id.item(),
                traj_tensor=seq_trajs.detach().cpu(),
                prob_tensor=seq_probs.detach().cpu(),
            )
            for seq_id, seq_trajs, seq_probs in zip(
                data.seq_id, traj_sliced_per_seq, prob_sliced_per_seq
            )
        ]
        
        return output

    def predict_step(  # type: ignore[override]
        self,
        data: TemporalData,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> List[SeqTrajectory]:
        # Forwarding the data through the model
        y_hat: Tensor
        pi: Tensor
        y_hat, pi = self(data)

        output = self.get_seqtrajectory_list(data, y_hat, pi)

        return output
