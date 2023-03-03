from typing import Dict, Tuple

import torch
from torch import Tensor

from models.hivt import HiVT
from utils import TemporalData


class HiVTSubmit(HiVT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_restore_rotation(self, data: Tensor, theta: Tensor) -> Tensor:
        assert data.shape[0] == theta.shape[0]
        batch_size = data.shape[0]
        rotation_mat = torch.empty(batch_size, 2, 2, device=data.device)
        sin_vals = torch.sin(theta)
        cos_vals = torch.cos(theta)
        rotation_mat[:, 0, 0] = cos_vals
        rotation_mat[:, 0, 1] = -sin_vals
        rotation_mat[:, 1, 0] = sin_vals
        rotation_mat[:, 1, 1] = cos_vals
        return torch.bmm(
            data.reshape(batch_size, -1, 2), rotation_mat.inverse()
        ).reshape(data.shape)

    def predict_step(
        self,
        data: TemporalData,
        batch_idx: int,
        dataloader_idx: int = None,
    ) -> Tuple[
        Dict[int, torch.Tensor], Dict[int, torch.Tensor]
    ]:  # pylint: disable=arguments-differ
        """
        Returns:
            Dict[seq_ids, predicted values]
        """
        # Forwarding the data through the model
        y_hat: torch.Tensor
        pi: torch.Tensor
        y_hat, pi = self(data)

        batch_size = data.seq_id.shape[0]

        # Info about the output data shape
        assert y_hat.shape == (
            self.num_modes,
            data.num_nodes,
            self.future_steps,
            4,
        )
        assert pi.shape == (data.num_nodes, self.num_modes)

        nodes_traj: Tensor = y_hat[:, :, :, :2].transpose(0, 1)
        softmax_pi: Tensor = torch.softmax(pi, dim=1)

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

        # Selecting data for agents
        predicted_val: torch.Tensor = self.batch_restore_rotation(
            nodes_traj[data.agent_index, :, :, :2]
            + data.positions[data.agent_index, 19, :]
            .unsqueeze(-2)
            .unsqueeze(-2)
            .expand(batch_size, self.num_modes, self.future_steps, -1),
            data.theta,
        ) + data.origin.unsqueeze(-2).unsqueeze(-2).expand(
            batch_size, self.num_modes, self.future_steps, 2
        )
        predicted_prob: torch.Tensor = softmax_pi[data.agent_index, :]

        assert predicted_val.shape == (
            len(data.agent_index),
            self.num_modes,
            self.future_steps,
            2,
        )
        assert predicted_prob.shape == (len(data.agent_index), self.num_modes)

        # Creating output dict
        assert data.seq_id.ndim == 1
        assert data.seq_id.shape[0] == len(data.agent_index)
        output_traj: Dict[int, torch.Tensor] = {
            seq_t.item(): pred_traj
            for seq_t, pred_traj in zip(data.seq_id, predicted_val)
        }

        output_prob: Dict[int, torch.Tensor] = {
            seq_t.item(): pred_prob
            for seq_t, pred_prob in zip(data.seq_id, predicted_prob)
        }

        return output_traj, output_prob
