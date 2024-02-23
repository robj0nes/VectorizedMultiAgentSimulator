import torch

from vmas.simulator.core import Agent, World, Landmark, Box
from vmas.simulator.utils import Color


class DOTSAgent(Agent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.payload = None


class DOTSWorld(World):
    def __init__(self, batch_dim, device, **kwargs):
        super().__init__(batch_dim, device, **kwargs)
        self.device = device
        self.arena_size = 5
        self.viewer_zoom = 1.7

    def spawn_map(self):
        self.walls = []
        for i in range(4):
            wall = Landmark(
                name=f"wall_{i}",
                collide=True,
                shape=Box(length=self.arena_size + 0.1, width=0.1),
                color=Color.BLACK
            )
            self.walls.append(wall)
            self.add_landmark(wall)

    def reset_map(self, env_index):
        # Align walls with the edge of the arena
        for i, landmark in enumerate(self.walls):
            landmark.set_pos(
                torch.tensor(
                    [
                        -self.arena_size / 2
                        if i == 0
                        else self.arena_size / 2
                        if i == 1
                        else 0,
                        -self.arena_size / 2
                        if i == 2
                        else self.arena_size / 2
                        if i == 3
                        else 0,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 2] if i < 2 else 0,
                    dtype=torch.float32,
                    device=self.device,
                ),
                batch_index=env_index,
            )
