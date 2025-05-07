import torch

from vmas.simulator.dynamics.common import Dynamics


class Grid(Dynamics):
    def __init__(self, action_size: int, step: float):
        super().__init__()
        self.action_size = action_size
        self.step = step

    @property
    def needed_action_size(self) -> int:
        return self.action_size

    def process_action(self):
        pass
    #     action = self.agent.action.u
    #     if not action.abs().sum().item() == 0 and not self.moving:
    #         print(action)
    #         # TODO: Compute z as a change in grid instance (ie. go up a level)
    #         z = action[:, 2].to(torch.int)
    #         self.agent.state.pos[:, 0] += self.step * action[:, 0].to(torch.int)
    #         self.agent.state.pos[:, 1] += self.step * action[:, 1].to(torch.int)
    #
