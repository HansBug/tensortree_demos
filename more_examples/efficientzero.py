from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import numpy as np
import torch
import treetensor.torch as ttorch

from ding.torch_utils import SGD
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.rl_utils import MCTS, Root
from .muzero import MuZeroPolicy, ModifiedCrossEntropyLoss


@POLICY_REGISTRY.register('efficient_zero')
class EfficientZeroPolicy(MuZeroPolicy):
    """
    Overview:
        EfficientZero
    """

    def _forward_learn(self, data: ttorch.Tensor) -> Dict[str, Union[float, int]]:
        self._learn_model.train()
        losses = ttorch.as_tensor({})
        losses.consistent_loss = torch.zeros(1).to(self.device)
        losses.value_prefix_loss = torch.zeros(1).to(self.device)

        # first step
        output = self._learn_model.forward(data.obs, mode='init')
        losses.value_loss = self._ce(output.value, data.target_value[0])
        td_error_per_sample = losses.value_loss.clone().detach()
        losses.policy_loss = self._ce(output.logit, data.target_action[0])

        # unroll N step
        N = self._cfg.image_unroll_len
        for i in range(N):
            output = self._learn_model.forward(
                output.hidden_state, output.hidden_state_reward, data.action[i], mode='recurrent'
            )
            losses.value_loss += self._ce(output.value, data.target_value[i + 1])
            losses.policy_loss += self._ce(output.logit, data.target_action[i + 1])
            # consistent loss
            with torch.no_grad():
                next_hidden_state = self._learn_model.forward(data.next_obs, mode='init').hidden_state
                projected_next = self._learn_model.forward(next_hidden_state, mode='project')
            projected_now = self._learn_model.forward(output.hidden_state, mode='project')
            losses.consistent_loss += -(self._cos(projected_now, projected_next) * data.mask[i])
            # value prefix loss
            losses.value_prefix_loss += self._ce(output.value_prefix, data.target_value_prefix[i])
            # set half gradient
            output.hidden_state.register_hook(lambda grad: grad * 0.5)
            # reset hidden states
            if (i + 1) % self._cfg.lstm_horizon_len == 0:
                output.hidden_state_reward.zero_()

        total_loss = (
            self._cfg.policy_weight * losses.policy_loss + self._cfg.value_weight * losses.value_loss +
            self._cfg.value_prefix_weight * losses.value_prefix_loss +
            self._cfg.consistent_weight * losses.consistent_loss
        )
        total_loss = total_loss.mean()
        total_loss.register_hook(lambda grad: grad / N)

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
        }.update({k: v.mean().item()
                  for k, v in losses.items()})


