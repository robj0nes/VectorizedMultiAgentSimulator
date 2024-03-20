import typing

import torch
from torch import Tensor

from vmas.simulator.core import Agent, World, Landmark, Box, AgentState, EntityState
from vmas.simulator.utils import Color, override


class DOTSWorld(World):
    def __init__(self, batch_dim, device, **kwargs):
        super().__init__(batch_dim, device, **kwargs)
        self.walls = None
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


# TODO: Define a default action script which accounts for the DOTs action space.
class DOTSAgent(Agent):
    def __init__(self, name, payload_shape=None, **kwargs):
        super().__init__(name, **kwargs)
        self.payload_shape = payload_shape
        self._state = DOTSAgentState(payload_shape)

    def set_payload(self, payload: Tensor, batch_index: int):
        self._set_state_property(DOTSAgentState.payload, self.state, payload, batch_index)


class DOTSAgentState(AgentState):
    def __init__(self, payload_shape=None):
        super().__init__()
        self.payload_shape = payload_shape

        # Has agent completed primary task and is now seeking goal.
        self._seeking_goal = None

        # Defines the agent payload
        self._payload = None

    @property
    def seeking_goal(self):
        return self._seeking_goal

    @seeking_goal.setter
    def seeking_goal(self, seeking_goal: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
                seeking_goal.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {seeking_goal.shape[0]}, expected {self._batch_dim}"

        self._seeking_goal = seeking_goal.to(self._device)

    @property
    def payload(self):
        return self._payload

    @payload.setter
    def payload(self, payload: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting"
        assert (
                payload.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {payload.shape[0]}, expected {self._batch_dim}"

        self._payload = payload.to(self._device)

    @override(AgentState)
    def _reset(self, env_index: typing.Optional[int]):
        if self.seeking_goal is not None:
            if env_index is None:
                self.seeking_goal[:] = False
            else:
                self.seeking_goal[env_index] = False

        if self.payload is not None:
            if env_index is None:
                self.payload[:] = 0
            else:
                self.payload[env_index] = 0

        super()._reset(env_index)

    @override(AgentState)
    def _spawn(self, dim_c: int, dim_p: int):
        self.seeking_goal = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.bool
        )
        if self.payload_shape is not None:
            self.payload = torch.zeros(
                self.batch_dim, self.payload_shape, device=self.device, dtype=torch.float32
            )
        super()._spawn(dim_c, dim_p)

class DOTSPayloadDest(Landmark):
    def __init__(self, expected_payload_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.expected_payload_shape = expected_payload_shape
        self._state = DOTSPayloadDestState(expected_payload_shape)

    def set_expected_payload(self, payload: Tensor, batch_index: int):
        self._set_state_property(DOTSPayloadDestState.expected_payload, self.state, payload, batch_index)


class DOTSPayloadDestState(EntityState):
    def __init__(self, expected_payload_shape=None):
        super().__init__()
        self.expected_payload_shape = expected_payload_shape
        self._expected_payload = None

    @property
    def expected_payload(self):
        return self._expected_payload

    @expected_payload.setter
    def expected_payload(self, exp_payload: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting"
        assert (
                exp_payload.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {exp_payload.shape[0]}, expected {self._batch_dim}"

        self._expected_payload = exp_payload.to(self._device)

    def _reset(self, env_index: typing.Optional[int]):
        if self.expected_payload is not None:
            if env_index is None:
                self.expected_payload[:] = 0
            else:
                self.expected_payload[env_index] = 0

        super()._reset(env_index)

    def _spawn(self, dim_c: int, dim_p: int):
        if self.expected_payload_shape is not None:
            self.expected_payload = torch.zeros(
                self.batch_dim, self.expected_payload_shape, device=self.device, dtype=torch.float32
            )
        super()._spawn(dim_c, dim_p)



