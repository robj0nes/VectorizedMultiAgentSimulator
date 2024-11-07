#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.gbp import GaussianBeliefPropagation
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box
from vmas.simulator.dots_core import DOTSGBPWorld, DOTSGBPAgent, DOTSGBPGoal
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar, ObjectDetectionSensor
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


# Note: Parking GBP Explorations for now. Needs more consideration to apply within this framework.
#   - Dynamic graph is important, but how can this be managed efficiently across parallel training envs?

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)
        self.use_gbp = kwargs.pop("use_gbp", False)

        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 1
        )  # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 1
        )  # Y-coordinate limit for entities spawning
        self.enforce_bounds = kwargs.pop(
            "enforce_bounds", False
        )  # If False, the world is unlimited; else, constrained by world_spawning_x and world_spawning_y.

        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None

        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        if self.split_goals:
            assert (
                    self.n_agents % 2 == 0
                    and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world
        if self.use_gbp:
            world = DOTSGBPWorld(batch_dim, device)
        else:
            world = World(batch_dim, device, substeps=2)

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            if self.use_gbp:
                pose_count = 2
                entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, DOTSGBPAgent)
                entity_filter_goals: Callable[[Entity], bool] = lambda e: isinstance(e, DOTSGBPGoal)
                # graph_dict = {
                #     'agents': {
                #         'nodes': [j for j in range(self.n_agents)],
                #         'edges': [[i, j] for j in range(self.n_agents) if i != j]
                #     },
                #     'goals': {
                #         'nodes': [j for j in range(self.n_agents, self.n_agents * 2)],
                #         'edges': [[i, j] for j in range(self.n_agents, self.n_agents * 2)]
                #     },
                #     # 'pose': {
                #     #     'nodes': [],
                #     #     'edges': []
                #     # }
                #     'pose': {
                #         'nodes': [j for j in range(self.n_agents * 2, self.n_agents * 2 + pose_count)],
                #         'edges': [[i, self.n_agents * 2]] + [[j, k] for j, k in zip(
                #             range(self.n_agents * 2, self.n_agents * 2 + pose_count - 1),
                #             range(self.n_agents * 2 + 1, self.n_agents * 2 + pose_count))],
                #     }
                # }
                graph_dict = {
                    'pose': {
                        'nodes': [0]
                    }

                }

                gbp = GaussianBeliefPropagation(graph_dict=graph_dict,
                                                msg_passing_iters=1,
                                                msgs_per_iter=1,
                                                batch_dim=batch_dim,
                                                device=device)
                # Note: For now assumes n_agents == n_goals
                agent = DOTSGBPAgent(
                    name=f"agent_{i}",
                    gbp=gbp,
                    collide=self.collisions,
                    color=color,
                    shape=Sphere(radius=self.agent_radius),
                    render_action=True,
                    world=world,
                    n_agents=self.n_agents,
                    n_goals=self.n_agents,
                    sensors=(
                        [
                            ObjectDetectionSensor(
                                world,
                                n_rays=16,
                                max_range=self.lidar_range,
                                entity_filter=entity_filter_agents,
                            ),
                            ObjectDetectionSensor(
                                world,
                                n_rays=16,
                                max_range=self.lidar_range,
                                entity_filter=entity_filter_goals,
                            )
                        ]
                        if self.collisions
                        else None
                    ),
                    tensor_args={"batch_dim": batch_dim, "device": device}
                )

            else:
                # Constraint: all agents have same action range and multiplier
                agent = Agent(
                    name=f"agent_{i}",
                    collide=self.collisions,
                    color=color,
                    shape=Sphere(radius=self.agent_radius),
                    render_action=True,
                    sensors=(
                        [
                            Lidar(
                                world,
                                n_rays=12,
                                max_range=self.lidar_range,
                                entity_filter=entity_filter_agents,
                            ),
                        ]
                        if self.collisions
                        else None
                    ),
                )

            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            if self.use_gbp:
                goal = DOTSGBPGoal(
                    name=f"goal {i}",
                    collide=False,
                    color=color
                )
            else:
                goal = Landmark(
                    name=f"goal {i}",
                    collide=False,
                    color=color,
                )
            world.add_landmark(goal)
            agent.goal = goal

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_spawning_x, self.world_spawning_x),
            (-self.world_spawning_y, self.world_spawning_y),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        for i, agent in enumerate(self.world.agents):
            if self.use_gbp:
                # Update our robot position anchor with the starting position.
                # agent.gbp.update_anchor(agent.state.pos, anchor_index=i, env_index=env_index)
                agent.gbp.update_anchor(agent.state.pos, anchor_index=0, env_index=env_index)

            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                        torch.linalg.vector_norm(
                            agent.state.pos - agent.goal.state.pos,
                            dim=1,
                        )
                        * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                        torch.linalg.vector_norm(
                            agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                        )
                        * self.pos_shaping_factor
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                            ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                            ] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    # Note: do we want to do pos-shaping in GBP verison?.. probably not!
    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        if self.use_gbp:
            # Process GBP msg passing before taking observations.
            self.world.update_and_iterate_gbp(agent)
            sensor_measurements = agent.sensors[0]._max_range - agent.sensors[0].measure()[0]

            goal_poses = []
            if self.observe_all_goals:
                for a in self.world.agents:
                    goal_poses.append(agent.state.pos - a.goal.state.pos)
            else:
                goal_poses.append(agent.state.pos - agent.goal.state.pos)
            return torch.cat(
                [
                    agent.state.pos,
                    agent.state.vel,
                ]
                + goal_poses
                + (
                    [sensor_measurements]
                    if self.collisions
                    else []
                ),
                dim=-1,
            )

            # sensor_measurements = agent.sensors[0]._max_range - agent.sensors[0].measure()[0]
            # goal_poses = []
            # own_pos_est = agent.gbp.current_means[:, agent.agent_index]
            # if self.observe_all_goals:
            #     for i in agent.gbp.graph_dict['goals']['nodes']:
            #         goal_pos_est = agent.gbp.current_means[:, i]
            #         goal_poses.append((own_pos_est - goal_pos_est).float())
            # else:
            #     # We assume the goal index is the same as the agent_index
            #     goal_index = agent.gbp.graph_dict['goals']['nodes'][agent.agent_index]
            #     goal_pos_est = agent.gbp.current_means[:, goal_index]
            #     goal_poses.append((own_pos_est - goal_pos_est).float())
            # return torch.cat(
            #     [
            #         own_pos_est.float(),
            #         agent.state.vel
            #     ]
            #     + goal_poses
            #     + (
            #         [sensor_measurements]
            #         if self.collisions
            #         else []
            #     ),
            #     dim=-1,
            # )

        else:
            sensor_measurements = agent.sensors[0]._max_range - agent.sensors[0].measure()

            goal_poses = []
            if self.observe_all_goals:
                for a in self.world.agents:
                    goal_poses.append(agent.state.pos - a.goal.state.pos)
            else:
                goal_poses.append(agent.state.pos - agent.goal.state.pos)
            return torch.cat(
                [
                    agent.state.pos,
                    agent.state.vel,
                ]
                + goal_poses
                + (
                    [sensor_measurements]
                    if self.collisions
                    else []
                ),
                dim=-1,
            )

    def done(self):
        return torch.stack(
            [

                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering
        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon=0.2, clf_slack=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon  # Exponential CLF convergence rate
        self.clf_slack = clf_slack  # weights on CLF-QP slack variable

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        QP inputs:
        These values need to computed apriri based on observation before passing into QP

        V: Lyapunov function value
        lfV: Lie derivative of Lyapunov function
        lgV: Lie derivative of Lyapunov function
        CLF_slack: CLF constraint slack variable

        QP outputs:
        u: action
        CLF_slack: CLF constraint slack variable, 0 if CLF constraint is satisfied
        """
        # Install it with: pip install cvxpylayers
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        goal_pos = (-1.0) * (observation[:, 4:6] - agent_pos)

        # Pre-compute tensors for the CLF and CBF constraints,
        # Lyapunov Function from: https://arxiv.org/pdf/1903.03692.pdf

        # Laypunov function
        V_value = (
                (agent_pos[:, X] - goal_pos[:, X]) ** 2
                + 0.5 * (agent_pos[:, X] - goal_pos[:, X]) * agent_vel[:, X]
                + agent_vel[:, X] ** 2
                + (agent_pos[:, Y] - goal_pos[:, Y]) ** 2
                + 0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) * agent_vel[:, Y]
                + agent_vel[:, Y] ** 2
        )

        LfV_val = (2 * (agent_pos[:, X] - goal_pos[:, X]) + agent_vel[:, X]) * (
            agent_vel[:, X]
        ) + (2 * (agent_pos[:, Y] - goal_pos[:, Y]) + agent_vel[:, Y]) * (
                      agent_vel[:, Y]
                  )
        LgV_vals = torch.stack(
            [
                0.5 * (agent_pos[:, X] - goal_pos[:, X]) + 2 * agent_vel[:, X],
                0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) + 2 * agent_vel[:, Y],
            ],
            dim=1,
        )
        # Define Quadratic Program (QP) based controller
        u = cp.Variable(2)
        V_param = cp.Parameter(1)  # Lyapunov Function: V(x): x -> R, dim: (1,1)
        lfV_param = cp.Parameter(1)
        lgV_params = cp.Parameter(
            2
        )  # Lie derivative of Lyapunov Function, dim: (1, action_dim)
        clf_slack = cp.Variable(1)  # CLF constraint slack variable, dim: (1,1)

        constraints = []

        # QP Cost F = u^T @ u + clf_slack**2
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack ** 2)

        # control bounds between u_range
        constraints += [u <= u_range]
        constraints += [u >= -u_range]
        # CLF constraint
        constraints += [
            lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack <= 0
        ]

        QP_problem = cp.Problem(qp_objective, constraints)

        # Initialize CVXPY layers
        QP_controller = CvxpyLayer(
            QP_problem,
            parameters=[V_param, lfV_param, lgV_params],
            variables=[u],
        )

        # Solve QP
        CVXpylayer_parameters = [
            V_value.unsqueeze(1),
            LfV_val.unsqueeze(1),
            LgV_vals,
        ]
        action = QP_controller(*CVXpylayer_parameters, solver_args={"max_iters": 500})[
            0
        ]

        return action


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=False,
        use_gbp=True,
        lidar_range=0.2,
        n_agents=3
    )
