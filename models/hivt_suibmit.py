from typing import Dict, Tuple

import torch
from torch import Tensor

from models.hivt import HiVT
from utils import TemporalData, visualize


class HiVTSubmit(HiVT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # Info about the output data shape
        assert y_hat.shape == (
            self.num_modes,
            data.num_nodes,
            self.future_steps,
            4,
        )
        assert pi.shape == (data.num_nodes, self.num_modes)

        nodes_traj: Tensor = y_hat[:, :, :, :2].transpose(0, 1)

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
        predicted_val: torch.Tensor = (
            visualize.restore_rotation(
                nodes_traj[data.agent_index, :, :, :2], data.theta
            )
            + data.origin.expand(6, 30, 2)
            - data.positions[data.agent_index, 19, :].expand(6, 30, -1)
        )
        predicted_prob: torch.Tensor = pi[data.agent_index, :]

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
