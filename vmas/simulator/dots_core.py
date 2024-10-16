import math
import time
import re
import numpy as np
from typing import TypedDict, List, Callable, Iterable, Optional, Union, Dict

import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict
from torch import Tensor
from sys import platform
import torch_bp.distributions as dist
from torch_bp.bp import LoopyLinearGaussianBP
from torch_bp.graph import FactorGraph
from torch_bp.graph.factors import UnaryFactor, PairwiseFactor
from torch_bp.graph.factors.linear_gaussian_factors import NaryGaussianLinearFactor, UnaryGaussianLinearFactor, \
    PairwiseGaussianLinearFactor

# Importing rendering breaks BC/BP clusters
if platform == "darwin":
    from vmas.simulator import rendering
    from vmas.simulator.rendering import Geom

from vmas.simulator.core import Agent, World, Landmark, Box, AgentState, EntityState
from vmas.simulator.utils import Color, override


class DOTSWorld(World):
    def __init__(self, batch_dim, device, **kwargs):
        super().__init__(batch_dim, device, **kwargs)
        self.arena_size = 5
        self.viewer_zoom = 1.7
        # TODO: Define physical constraints for all DOTS arena implementations.


class DOTSPaintingWorld(DOTSWorld):
    def __init__(self, batch_dim, device, **kwargs):
        super().__init__(batch_dim, device, **kwargs)
        self.walls = None
        self.device = device

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


class DOTSAgent(Agent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.agent_index = int(re.findall(r'\d+', self.name)[0])
    # TODO: Define physical constraints for any DOTS robot implementation.


class DOTSGBPAgent(DOTSAgent):
    def __init__(self, name, gbp, world, n_agents, n_goals, tensor_args, **kwargs):
        super().__init__(name, **kwargs)
        self.device = tensor_args['device']
        self.gbp = gbp
        self.world = world
        self.n_agents = n_agents
        self.n_goals = n_goals

    # TODO: Dont use self.state.pos - instead take our current position estimate and add the action vector
    def update_own_position_estimate(self):
        if self.action.u is not None:
            if torch.count_nonzero(self.action.u) > 0:
                self.gbp.update_anchor(x=self.state.pos, anchor_index=self.agent_index)


    def estimate_goal_pos_from_sensors(self):
        if self.gbp.current_means is not None:
            distances, entities = self.sensors[1].measure()
            for i in range(self.n_goals):
                entity_mask = i+1
                dists = distances[entities == entity_mask]
                if dists.numel() > 0:
                    # Get all pairs of [batch_dim, ray_index] for any collisions with the goal.
                    detections = torch.nonzero(entities == entity_mask).squeeze()
                    self.update_entity_anchor(detections=detections,
                                              dists=dists,
                                              anchor_index=self.gbp.graph_dict['goals']['nodes'][i])

    def estimate_other_agent_pos_from_sensors(self):
        if self.gbp.current_means is not None:
            # NOTE: We need to be concious of the return structure of `entities`:
            #   '0' represents no entity detection.
            #   i > 0 represents the ith entity checked when computing ray collisions
            #   therefore if we are checking for agent 1:
            #       '1' = agent 0, '2' = agent 2, '3' = agent 3 etc.
            #       In general: entries are indexed i + 1 if i < self.agent_index else i
            distances, entities = self.sensors[0].measure()
            # Go through all agents != self
            for i in self.gbp.graph_dict['agents']['nodes']:
            # for i in range(self.n_agents):
                if i != self.agent_index:
                    # Adjust agent index according to notes above
                    entity_mask = i + 1 if i < self.agent_index else i
                    # Find all distances related to the other agent index
                    dists = distances[entities == entity_mask]
                    if dists.numel() > 0:
                        # Get all pairs of [batch_dim, ray_index] for any collisions with the other agent.
                        detections = torch.nonzero(entities == entity_mask).squeeze()
                        self.update_entity_anchor(detections=detections,
                                                  dists=dists,
                                                  anchor_index=i)

    def update_entity_anchor(self, detections, dists, anchor_index):
        if len(detections.shape) == 1:
            detections = detections.unsqueeze(0)
        # NOTE: Not optimised for tensors.. rethink when there's some time to look at it.
        # Find the minimum sensor distance to the other agent in each detected batch dimension.
        batch_dims = detections[:, 0]
        for env_index in batch_dims.unique():
            batch_mask = (batch_dims == env_index)
            batch_dists = dists[batch_mask]
            min_dist, min_index = torch.abs(batch_dists).min(dim=0)
            ray_index = detections[batch_mask][min_index].squeeze()[-1]
            # Using current estimate of own position, estimate the position of the other agent
            own_pos_est = self.gbp.current_means[env_index, self.agent_index]
            new_pos_est = torch.stack(
                [
                    own_pos_est[0] + min_dist * torch.cos(self.sensors[1]._angles[env_index][ray_index]),
                    own_pos_est[1] + min_dist * torch.sin(self.sensors[1]._angles[env_index][ray_index])
                ],
                dim=-1
            ).to(self.device)
            self.gbp.update_anchor(x=new_pos_est, anchor_index=anchor_index, env_index=env_index)

    def render(self, env_index: int = 0, selected_agents: List[int] = None, show_gaussians: bool = True) -> "List[Geom]":
        geoms = super().render(env_index)
        if not show_gaussians:
            return geoms

        # Selected agents is not None if we are rendering interactively. Otherwise we render agent 0 by default.
        if (selected_agents is not None and self.agent_index in selected_agents
                or selected_agents is None and '0' in self.name):
            # Create Gaussian distributions from all variables.
            gaussians = [dist.Gaussian(mu, sigma, device=self.device)
                         for mu, sigma in zip(self.gbp.current_means[env_index], self.gbp.current_covars[env_index])]

            for i, g in enumerate(gaussians):
                # Note: Assuming n_agents == n_goals
                entity_index = i % self.n_agents

                # Eval gaussian grid-wise in worldspace and collect any pdf(x) > 2
                np.set_printoptions(legacy='1.25')
                X, Y, Z = g.eval_grid([-self.world.arena_size, self.world.arena_size,
                                       -self.world.arena_size, self.world.arena_size],
                                      n_samples=200)
                ys, xs = np.where(Z > 0.5)
                # Extract the world coordinates at each evaulation point. TODO: More thorough testing that this is correct.
                marker_pos = [(X[0][xs[i]], Y[ys[i]][0], Z[ys[i]][xs[i]]) for i in range(len(xs))]

                # mean_loc = np.where(Z == np.max(Z))
                # mean_pos = (X[0][mean_loc[1][0]], Y[mean_loc[0][0]][0], Z[mean_loc[0][0]][mean_loc[1][0]])
                if len(marker_pos) > 0:
                    # Normalise the z values for better rendering.
                    zs = [m[2] for m in marker_pos]
                    min_zs = np.min(zs)
                    max_zs = np.max(zs)
                    norm_markers = [(mp[0], mp[1], (mp[2] - min_zs) / (max_zs - min_zs)) for mp in marker_pos]

                    # norm_markers = marker_pos
                    for m in norm_markers:
                        marker = rendering.make_circle(radius=0.01, filled=True)
                        marker.set_color(*self.world.agents[entity_index].color, alpha=m[2] - 0.5)
                        marker_xform = rendering.Transform()
                        marker_xform.set_translation(m[0], m[1])
                        marker.add_attr(marker_xform)
                        geoms.append(marker)
        return geoms


class DOTSGBPWorld(DOTSWorld):
    def __init__(self, batch_dim, device, **kwargs):
        super().__init__(batch_dim, device, **kwargs)

    def update_and_iterate_gbp(self, agent: DOTSGBPAgent):
        agent.update_own_position_estimate()
        agent.estimate_other_agent_pos_from_sensors()
        agent.estimate_goal_pos_from_sensors()
        agent.gbp.iterate_gbp()

# Currently used to distinguish goal landmarks with Lidar filters.
class DOTSGBPGoal(Landmark):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)


class DOTSPaintingAgent(DOTSAgent):
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
    def __init__(self, agent_index, knowledge_shape=None):
        super().__init__()
        self.agent_index = agent_index
        self.knowledge_shape = knowledge_shape

        self._reward_multiplier = None

        # Has agent completed primary task and is now seeking goal.
        self._task_complete = None
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
    def _reset(self, env_index: Optional[int]):
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

    def _reset(self, env_index: Optional[int]):
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
