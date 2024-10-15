import math
import time
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
    # TODO: Define physical constraints for any DOTS robot implementation.


class DOTSGBPAgent(DOTSAgent):
    def __init__(self, name, tensor_args, **kwargs):
        super().__init__(name, **kwargs)
        self.device = tensor_args['device']
        self.total_nodes = None
        self.factors = None
        self.factor_neighbours = None
        self.init_mu = None
        self.init_covar = None
        self.means = None
        self.covars = None
        self.gbp = None

    def unary_anchor_fn(self, x: torch.Tensor):
        batch_shape = x.shape[:1]
        grad = torch.eye(x.shape[-1], dtype=x.dtype).repeat(batch_shape + (1, 1))
        return grad, x

    def pairwise_dist_fn(self, x: torch.Tensor):
        h_fn = lambda x: torch.linalg.norm(x[:, 2:] - x[:, :2], dim=-1)
        grad_fn = torch.func.jacrev(h_fn)
        grad_x = grad_fn(x).diagonal(dim1=0, dim2=1).transpose(1, 0)
        return grad_x[:, None, :], h_fn(x).unsqueeze(-1)

    def init_factor_graph(self, n_agents, n_goals):
        """Initialise a base FG to assign to all robots. This implementation defines factors between
        robots own positions, other robot positions, and goal positions."""
        # Note:
        #   - We fix the total variables (nodes) for the factor graph according to the task needs
        #   - Initialise the graph with 'zero' factors between known relationships
        #   - As the task evolves we introduce factors where necessary,
        #       but must be introduced across all batched envs (batch_dim)
        pvn_count = 1  # position variable nodes.
        gvn_count = 0  # goal variable nodes
        rvn_count = n_agents - 1  # robot variable nodes
        self.total_nodes = pvn_count + gvn_count + rvn_count

        sigma = 0.05
        self.init_node_mu = (torch.randn(self.batch_dim, self.total_nodes, 2) * 2 + 2).to(self.device, dtype=torch.float64)
        self.init_node_covar = (torch.eye(2).repeat(self.batch_dim, self.total_nodes, 1, 1) * 2).to(self.device, dtype=torch.float64)
        factor_neighbours = []

        anchor_factors = [UnaryGaussianLinearFactor(self.unary_anchor_fn,
                                                       # Get position
                                                       self.state.pos if i == 0 else torch.zeros_like(self.state.pos),
                                                       sigma * torch.eye(2, device=self.device, dtype=torch.float64)
                                                            .unsqueeze(0).repeat(self.batch_dim, 1, 1),
                                                       self.init_node_mu[:, 0].to(self.device),
                                                       True)
                          for i in range(self.total_nodes)]
        factor_neighbours.extend([(i,) for i in range(self.total_nodes)])


        # initial z_bias: [batch_dim, 1], covar: [batch_dim, 1, 1]
        self.init_dist_z_bias = (1.0 * torch.ones(1, device=self.device, dtype=torch.float64)
                                 .unsqueeze(0).repeat(self.batch_dim, 1))
        self.init_dist_covar = (2 * sigma * torch.eye(1, device=self.device, dtype=torch.float64)
                                .unsqueeze(0).repeat(self.batch_dim, 1, 1))

        # x_0 is a concatenated tensor of shape [batch_dim, mu_i + mu_j]
        # Position factors are the `pvn_count` most recent position estimates:
        position_factors = [PairwiseGaussianLinearFactor(self.pairwise_dist_fn,
                                                         self.init_dist_z_bias,
                                                         self.init_dist_covar,
                                                         torch.concat(
                                                             (self.init_node_mu[:, i], self.init_node_mu[:, j]),
                                                             dim=-1).to(self.device),
                                                         False)
                            for i, j in zip(range(pvn_count - 1), range(1, pvn_count))]
        factor_neighbours.extend((n, n + 1) for n in range(pvn_count - 1))

        # Goal factors exist for the distance estimate between the robots most recent position and the goal:
        #   factors:[pvn:pvn+gvn]
        goal_factors = [PairwiseGaussianLinearFactor(self.pairwise_dist_fn,
                                                     self.init_dist_z_bias,
                                                     self.init_dist_covar,
                                                     torch.concat(
                                                         (self.init_node_mu[:, 0], self.init_node_mu[:, i])
                                                         , dim=-1).to(self.device),
                                                     False)
                        for i in range(pvn_count, pvn_count + gvn_count)]
        factor_neighbours.extend((0, n) for n in range(pvn_count,
                                                       pvn_count + gvn_count))

        # Robot factors exist for the distance between a neighbouring robot and the most recent position:
        #   factors:[pvn+gvn:pvn+gvn+rvn]
        robot_factors = [PairwiseGaussianLinearFactor(self.pairwise_dist_fn,
                                                      self.init_dist_z_bias,
                                                      self.init_dist_covar,
                                                      torch.concat(
                                                          (self.init_node_mu[:, 0], self.init_node_mu[:, i]),
                                                          dim=-1).to(self.device),
                                                      False)
                         for i in range(pvn_count + gvn_count, self.total_nodes)]


        factor_neighbours.extend((0, n) for n in range(pvn_count + gvn_count,
                                                       self.total_nodes))

        self.factors = anchor_factors + position_factors + goal_factors + robot_factors
        self.factor_neighbours = factor_neighbours
        self.gbp = self.initialise_gbp()
        self.iterate_gbp()

    def initialise_gbp(self) -> LoopyLinearGaussianBP:
        fac_grap = FactorGraph(num_nodes=self.total_nodes,
                               factors=self.factors,
                               factor_neighbours=self.factor_neighbours)

        return LoopyLinearGaussianBP(node_means=self.init_node_mu, node_covars=self.init_node_covar, factor_graph=fac_grap,
                                     tensor_kwargs={'device': self.device,
                                                    'dtype': torch.float64},
                                     batch_dim=self.batch_dim)

    def iterate_gbp(self, num_iters=1, msg_pass_per_iter=1):
        self.means, self.covars = self.gbp.solve(num_iters=num_iters, msg_pass_per_iter=msg_pass_per_iter)
        self.vars = torch.diagonal(self.covars, dim1=-2, dim2=-1)
        self.stds = torch.sqrt(self.vars)
        # Note: position estimates are just sampled from the current node mean aod covars.. not very reliable??
        self.position_estimates = torch.normal(self.means, self.stds)


    def estimate_other_agent_pos_from_lidar(self):
        agent_detections = (self.sensors[0]._max_range - self.sensors[0].measure()).to(self.device)
        own_pos_est = self.position_estimates[:, 0]
        other_agent_xs = (own_pos_est[:, 0] + agent_detections * torch.cos(self.sensors[0]._angles)).to(self.device)
        other_agent_ys = (own_pos_est[:, 1] + agent_detections * torch.sin(self.sensors[0]._angles)).to(self.device)
        other_agent_positions = torch.stack((other_agent_xs, other_agent_ys), dim=-1)
        self.other_agent_positions = other_agent_positions[agent_detections > 0]

    def render(self, env_index: int = 0) -> "List[Geom]":
        geoms = super().render(env_index)
        if '0' in self.name:
            # for means, covars in zip(self.means[env_index], self.covars[env_index]):
            gaussians = [dist.Gaussian(mu, sigma, device=self.device) for mu, sigma in zip(self.means[env_index], self.covars[env_index])]
            for i, g in enumerate(gaussians):
                # eval gaussian in worldspace TODO: Get arena dims.
                X, Y, Z = g.eval_grid([-2, 2, -2, 2], n_samples=200)
                locs = np.where(Z > 0.5)
                marker_pos = [(Y[locs[1][i]][0], X[0][locs[0][i]], Z[locs[0][i]][locs[1][i]]) for i in range(len(locs[0]))]
                if len(marker_pos) > 0:
                    zs = [m[2] for m in marker_pos]
                    min_zs = np.min(zs)
                    max_zs = np.max(zs)
                    norm_markers = [(mp[0], mp[1], (mp[2] - min_zs) / (max_zs - min_zs)) for mp in marker_pos]
                    for m in norm_markers:
                        marker = rendering.make_circle(radius=0.005, filled=True)
                        marker.set_color(*self.color, alpha=m[2] - 0.25)
                        marker_xform = rendering.Transform()
                        marker_xform.set_translation(m[0], m[1])
                        marker.add_attr(marker_xform)
                        geoms.append(marker)
        return geoms


class DOTSGBPWorld(DOTSWorld):
    def __init__(self, batch_dim, device, **kwargs):
        super().__init__(batch_dim, device, **kwargs)

    def update_and_iterate_gbp(self, agent: DOTSGBPAgent):

        agent.estimate_other_agent_pos_from_lidar()
        # TODO: Update GBP nodes according to detected agent position estimates..
        #  Need to work out how to identify which agent we've detected..!

        agent.iterate_gbp(num_iters=1, msg_pass_per_iter=1)

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
