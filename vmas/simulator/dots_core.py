import math
import typing
from typing import List

import torch
from torch import Tensor
from sys import platform

# Importing rendering breaks BC/BP clusters
if platform == "darwin":
    from vmas.simulator import rendering
    from vmas.simulator.rendering import Geom

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


class DOTSAgent(Agent):
    def __init__(self, name, task, agent_index, render=True, knowledge_shape=None, **kwargs):
        super().__init__(name, **kwargs)
        self.rewards = dict()
        self.task = task
        self.agent_index = agent_index
        self.render_agent = render
        self.knowledge_shape = knowledge_shape
        self._counter_part = None
        self._state = DOTSAgentState(agent_index, knowledge_shape)

    @property
    def counter_part(self):
        return self._counter_part

    @counter_part.setter
    def counter_part(self, counter_part):
        self._counter_part = counter_part

    def set_knowledge(self, knowledge: Tensor, batch_index: int):
        self._set_state_property(DOTSAgentState.knowledge, self.state, knowledge, batch_index)

    @override(Agent)
    def render(self, env_index: int = 0) -> "List[Geom]":
        # This should only be called if the agent head is navigation..
        if self.render_agent:
            # TODO: Add agent labels?
            geoms = super().render(env_index)

            primary_knowledge = rendering.make_circle(angle=math.pi, radius=self.shape.radius / 2)
            mixed_knowledge = rendering.make_circle(angle=math.pi, radius=self.shape.radius / 2)

            if self.counter_part is not None:
                mix_head = [self.counter_part[i] for i in range(len(self.counter_part))
                            if self.counter_part[i].task == "listen"][0]
            else:
                mix_head = self
            primary_col = mix_head.state.knowledge[env_index][0].reshape(-1)[:3]
            mixed_col = mix_head.state.knowledge[env_index][1].reshape(-1)[:3]
            # Add a yellow ring around agents who have successfully matched their knowledge.
            if mix_head.state.task_complete[env_index]:
                success_ring = rendering.make_circle(radius=self.shape.radius, filled=True)
                success_ring.set_color(1, 1, 0)
                s_xform = rendering.Transform()
                s_xform.set_translation(self.state.pos[env_index][0], self.state.pos[env_index][1])
                success_ring.add_attr(s_xform)
                geoms.append(success_ring)

            primary_knowledge.set_color(*primary_col)
            mixed_knowledge.set_color(*mixed_col)
            p_xform = rendering.Transform()
            primary_knowledge.add_attr(p_xform)
            m_xform = rendering.Transform()
            mixed_knowledge.add_attr(m_xform)
            p_xform.set_translation(self.state.pos[env_index][0], self.state.pos[env_index][1])
            p_xform.set_rotation(math.pi)
            m_xform.set_translation(self.state.pos[env_index][0], self.state.pos[env_index][1])

            # label = TextLine(f"Agent {i}", x=agent.state.pos[env_index][X], y=agent.state.pos[env_index][Y] - 10)
            # geoms.append(label)
            geoms.append(primary_knowledge)
            geoms.append(mixed_knowledge)
        else:
            geoms = []
        return geoms


class DOTSAgentState(AgentState):
    def __init__(self, agent_index, knowledge_shape=None):
        super().__init__()
        self.agent_index = agent_index
        self.knowledge_shape = knowledge_shape

        self._reward_multiplier = None

        # Has agent completed primary task and is now seeking goal.
        self._task_complete = None

        # Has agent completed primary task and is now seeking goal.
        self._target_goal_index = None

        # Defines the agent knowledge(s)
        self._knowledge = None


    @property
    def reward_multiplier(self):
        return self._reward_multiplier

    @reward_multiplier.setter
    def reward_multiplier(self, reward_multiplier: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
                reward_multiplier.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {reward_multiplier.shape[0]}, expected {self._batch_dim}"

        self._reward_multiplier = reward_multiplier.to(self._device)


    @property
    def target_goal_index(self):
        return self._target_goal_index

    @target_goal_index.setter
    def target_goal_index(self, target_goal_index: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
                target_goal_index.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {target_goal_index.shape[0]}, expected {self._batch_dim}"

        self._target_goal_index = target_goal_index.to(self._device)

    @property
    def task_complete(self):
        return self._task_complete

    @task_complete.setter
    def task_complete(self, task_complete: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
                task_complete.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {task_complete.shape[0]}, expected {self._batch_dim}"

        self._task_complete = task_complete.to(self._device)

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
        if self.task_complete is not None:
            if env_index is None:
                self.task_complete[:] = False
            else:
                self.task_complete[env_index] = False

        if self.knowledge is not None:
            if env_index is None:
                self.knowledge[:] = 0
            else:
                self.knowledge[env_index] = 0

        if self.target_goal_index is not None:
            if env_index is None:
                self.target_goal_index[:] = self.agent_index
            else:
                self.target_goal_index[env_index] = self.agent_index

        if self.reward_multiplier is not None:
            if env_index is None:
                self.reward_multiplier[:] = 1
            else:
                self.reward_multiplier[env_index] = 1

        super()._reset(env_index)

    @override(AgentState)
    def _spawn(self, dim_c: int, dim_p: int):
        self.task_complete = torch.zeros(self.batch_dim, device=self.device, dtype=torch.bool)
        self.target_goal_index = torch.zeros(self.batch_dim, device=self.device, dtype=torch.long)
        self.reward_multiplier = torch.zeros(self.batch_dim, device=self.device, dtype=torch.int)

        if self.knowledge_shape is not None:
            self.knowledge = torch.zeros(
                self.batch_dim, *self.knowledge_shape, device=self.device, dtype=torch.float32
            )

        super()._spawn(dim_c, dim_p)


class DOTSPayloadDest(Landmark):
    def __init__(self, render=True, expected_knowledge_shape=None, position=None, **kwargs):
        super().__init__(**kwargs)
        self.render_goal = render
        self.position = position
        self.expected_knowledge_shape = expected_knowledge_shape
        self._state = DOTSPayloadDestState(expected_knowledge_shape)

    def set_expected_knowledge(self, knowledge: Tensor, batch_index: int):
        self._set_state_property(DOTSPayloadDestState.expected_knowledge, self.state, knowledge, batch_index)

    @override(Landmark)
    def render(self, env_index: int = 0) -> "List[Geom]":
        if self.render_goal:
            # TODO: Render additional actions here eg. print mixing coefficients.
            geoms = super().render(env_index)
            col = self.state.expected_knowledge[env_index]
            alpha = torch.tensor(
                [1 if self.state.solved[env_index] else 0.3], device=self.device, dtype=torch.float32)
            col = torch.cat([col, alpha], dim=0)
            geoms[0].set_color(*col)
        else:
            geoms = []
        return geoms


class DOTSPayloadDestState(EntityState):
    def __init__(self, expected_knowledge_shape=None):
        super().__init__()
        self.expected_knowledge_shape = expected_knowledge_shape
        self._expected_knowledge = None
        self._solved = None

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

    @property
    def solved(self):
        return self._solved

    @solved.setter
    def solved(self, solved: Tensor):
        assert (
                self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting"
        assert (
                solved.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {solved.shape[0]}, expected {self._batch_dim}"

        self._solved = solved.to(self._device)

    def _reset(self, env_index: typing.Optional[int]):
        if self.expected_knowledge is not None:
            if env_index is None:
                self.expected_knowledge[:] = 0
            else:
                self.expected_knowledge[env_index] = 0

        if self.solved is not None:
            if env_index is None:
                self.solved[:] = False
            else:
                self.solved[env_index] = False

        super()._reset(env_index)

    def _spawn(self, dim_c: int, dim_p: int):
        if self.expected_knowledge_shape is not None:
            self.expected_knowledge = torch.zeros(
                self.batch_dim, self.expected_knowledge_shape, device=self.device, dtype=torch.float32
            )

        self.solved = torch.zeros(self.batch_dim, device=self.device, dtype=torch.bool)
        super()._spawn(dim_c, dim_p)
