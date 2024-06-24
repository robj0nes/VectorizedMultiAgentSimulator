import typing
from typing import List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, World, Landmark, Box, AgentState, EntityState
from vmas.simulator.rendering import Geom
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
    def __init__(self, name, task, render=True, knowledge_shape=None, **kwargs):
        super().__init__(name, **kwargs)
        self.rewards = dict()
        self.task = task
        self.render_agent = render
        self.knowledge_shape = knowledge_shape
        self._state = DOTSAgentState(knowledge_shape)

    def set_knowledge(self, knowledge: Tensor, batch_index: int):
        self._set_state_property(DOTSAgentState.knowledge, self.state, knowledge, batch_index)

    @override(Agent)
    def render(self, env_index: int = 0) -> "List[Geom]":
        if self.render_agent:
            # TODO: Render additional actions here eg. print mixing coefficients.
            geoms = super().render(env_index)
        else:
            geoms = []
        return geoms


class DOTSComsNetwork(Agent):
    """
    Defines a separate 'agent' to represent an isolated communications network in the DOTS environment.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    @override(Agent)
    def render(self, env_index: int = 0) -> "List[Geom]":
        geoms = []
        return geoms


class DOTSAgentState(AgentState):
    def __init__(self, knowledge_shape=None):
        super().__init__()
        self.knowledge_shape = knowledge_shape

        # Has agent completed primary task and is now seeking goal.
        self._seeking_goal = None

        # Defines the agent knowledge(s)
        self._knowledge = None

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
    def knowledge(self):
        return self._knowledge

    @knowledge.setter
    def knowledge(self, knowledge: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting"
        assert (
                knowledge.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {knowledge.shape[0]}, expected {self._batch_dim}"

        self._knowledge = knowledge.to(self._device)

    @override(AgentState)
    def _reset(self, env_index: typing.Optional[int]):
        if self.seeking_goal is not None:
            if env_index is None:
                self.seeking_goal[:] = False
            else:
                self.seeking_goal[env_index] = False

        if self.knowledge is not None:
            if env_index is None:
                self.knowledge[:] = 0
            else:
                self.knowledge[env_index] = 0

        super()._reset(env_index)

    @override(AgentState)
    def _spawn(self, dim_c: int, dim_p: int):
        self.seeking_goal = torch.zeros(
            self.batch_dim, device=self.device, dtype=torch.bool
        )
        if self.knowledge_shape is not None:
            self.knowledge = torch.zeros(
                self.batch_dim, *self.knowledge_shape, device=self.device, dtype=torch.float32
            )
        super()._spawn(dim_c, dim_p)


class DOTSPayloadDest(Landmark):
    def __init__(self, expected_knowledge_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.expected_knowledge_shape = expected_knowledge_shape
        self._state = DOTSPayloadDestState(expected_knowledge_shape)

    def set_expected_knowledge(self, knowledge: Tensor, batch_index: int):
        self._set_state_property(DOTSPayloadDestState.expected_knowledge, self.state, knowledge, batch_index)


class DOTSPayloadDestState(EntityState):
    def __init__(self, expected_knowledge_shape=None):
        super().__init__()
        self.expected_knowledge_shape = expected_knowledge_shape
        self._expected_knowledge = None

    @property
    def expected_knowledge(self):
        return self._expected_knowledge

    @expected_knowledge.setter
    def expected_knowledge(self, exp_knowledge: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting"
        assert (
                exp_knowledge.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {exp_knowledge.shape[0]}, expected {self._batch_dim}"

        self._expected_knowledge = exp_knowledge.to(self._device)

    def _reset(self, env_index: typing.Optional[int]):
        if self.expected_knowledge is not None:
            if env_index is None:
                self.expected_knowledge[:] = 0
            else:
                self.expected_knowledge[env_index] = 0

        super()._reset(env_index)

    def _spawn(self, dim_c: int, dim_p: int):
        if self.expected_knowledge_shape is not None:
            self.expected_knowledge = torch.zeros(
                self.batch_dim, self.expected_knowledge_shape, device=self.device, dtype=torch.float32
            )
        super()._spawn(dim_c, dim_p)
