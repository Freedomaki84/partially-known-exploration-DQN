# double_dqn_sb3.py
from __future__ import annotations

import numpy as np
import torch as th
from torch.nn import functional as F
from typing import TypeVar

from stable_baselines3.dqn.dqn import DQN as SB3DQN

SelfDQN = TypeVar("SelfDQN", bound="DoubleDQN")


class DoubleDQN(SB3DQN):

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # type: ignore[override]
        # Train mode (affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # LR schedule step
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        for _ in range(gradient_steps):
            # Sample a minibatch from replay
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env) 
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # -------- Double DQN target --------
                # Select action with ONLINE net
                next_q_online = self.q_net(replay_data.next_observations)             # [B, A]
                next_actions = next_q_online.argmax(dim=1, keepdim=True)              # [B, 1]
                # Evaluate that action with TARGET net
                next_q_target = self.q_net_target(replay_data.next_observations)      # [B, A]
                next_q_values = next_q_target.gather(dim=1, index=next_actions)       # [B, 1]

                # Bellman target y = r + (1-d)*γ^n*Q_target(s', a*)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )  # [B, 1]

            # Current Q(s,a) from ONLINE net
            current_q_values = self.q_net(replay_data.observations)                   # [B, A]
            current_q_values = th.gather(current_q_values, 1, replay_data.actions.long())  # [B, 1]

            # Huber loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Logging
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))
