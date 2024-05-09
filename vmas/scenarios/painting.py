import math
import random
from enum import Enum

import numpy as np
import torch
import seaborn as sns

from vmas import render_interactively
from vmas.simulator.core import Sphere, World, Box
from vmas.simulator.dots_core import DOTSWorld, DOTSAgent, DOTSPayloadDest
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_REWARD_TYPE, AGENT_OBS_TYPE, ScenarioUtils, Y, X, Color


# Question:
#   1. How can I define agent action space
#   2. Rewards: Advice on how to form well structured reward signals...
#   3. Extending the evaluation video length.
#   4. Nan actions occurring sometimes?

def get_distinct_color_list(n_cols, device: torch.device, exclusions=None):
    opts = sns.color_palette(palette="Set2")
    return torch.stack([torch.tensor(opts[x], device=device, dtype=torch.float32) for x in
                        random.sample(range(0, len(opts)), n_cols)])


class TaskType(Enum):
    NAV = 0
    MIX = 1
    FULL = 2

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.observe_other_agents = None
        self.observe_all_goals = None
        self.rew = None
        self.goal_col_obs = None
        self.arena_size = None
        self.n_goals = None
        self.agent_radius = None
        self.n_agents = None
        self.goals = None
        self.agents_on_goal = None
        self.task_complexity = None
        self.communication_size = None

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = kwargs.get("n_agents", 4)
        self.agent_radius = 0.2
        self.payload_shape = (2, 3)

        self.n_goals = kwargs.get("n_goals", 4)
        self.observe_all_goals = kwargs.get("observe_all_goals", False)
        self.observe_other_agents = kwargs.get("observe_other_agents", True)
        self.communication_size = kwargs.get("communication_size", 0)

        self.task_complexity = kwargs.get("complexity", "nav")

        self.arena_size = 5
        self.viewer_zoom = 1.7

        # dim_c is the size of the communications signal.
        world = DOTSWorld(batch_dim, device, collision_force=100, dim_c=self.communication_size)
        # world = DOTSWorld(batch_dim, device, collision_force=100)

        for i in range(self.n_agents):
            agent = DOTSAgent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
                color=Color.GREEN,
                payload_shape=self.payload_shape,
                # Question: What does silent=False (ie. allowing comms) specifically do?
                silent=True if self.communication_size == 0 else False,
                # TODO: Make action size > 2 for mixing operations.
                # action_size=3
            )
            world.add_agent(agent)
            agent.agent_collision_rew = torch.zeros(batch_dim, device=device)
            agent.obstacle_collision_rew = agent.agent_collision_rew.clone()
            agent.agent_pos_reward = agent.agent_collision_rew.clone()
            agent.agent_final_reward = agent.agent_collision_rew.clone()

        self.goals = []
        for i in range(self.n_goals):
            goal = DOTSPayloadDest(
                name=f"goal_{i}",
                collide=False,
                shape=Box(length=self.agent_radius * 4, width=self.agent_radius * 4),
                color=Color.BLUE,
                expected_payload_shape=3
            )
            self.goals.append(goal)
            world.add_landmark(goal)

        world.spawn_map()

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -0.2)
        self.wall_collision_penalty = kwargs.get("passage_collision_penalty", -0.2)
        self.pos_shaping = kwargs.get("pos_shaping", False)
        self.pos_shaping_factor = kwargs.get("position_shaping_factor", 1.0)
        self.final_reward = kwargs.get("final_reward", 0.01)
        self.min_collision_distance = 0.005

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.final_rew = self.pos_rew.clone()
        return world

    def premix_paints(self, small, large, device):
        large_payloads = torch.stack(
            [
                get_distinct_color_list(large, device) for _ in range(self.world.batch_dim)
            ]
        )

        ran_ints = [
            random.sample(range(large_payloads.shape[1]), small) for _ in range(self.world.batch_dim)
        ]

        small_payloads = torch.stack(
            [
                torch.stack(
                    [
                        large_payloads[j][i] for i in ran_ints[j]
                    ]
                )
                for j in range(self.world.batch_dim)
            ]
        )

        for i in range(self.world.batch_dim):
            for sp in small_payloads[i]:
                assert (
                        sp in large_payloads[i]
                ), "Trying to assign matching payloads to goals, but found a mismatch"

        return small_payloads, large_payloads

    def unmixed_paints(self, device):
        t = np.linspace(-510, 510, self.n_agents)
        # Generate a linear distribution across RGB colour space
        agent_payloads = torch.stack(
            [
                torch.tensor(
                    np.round(
                        np.clip(
                            np.stack(

                                [-t, 510 - np.abs(t), t],
                                axis=1
                            ),
                            0,
                            255
                        ).astype(np.uint8)
                    ),
                    device=device)
                for _ in range(self.world.batch_dim)
            ]
        )

        # Generate a random colour in RGB colour space for the goals.
        goal_payloads = torch.stack(
            [
                torch.stack([
                    torch.tensor(random.sample(range(256), 3), device=device, dtype=torch.uint8)
                    for _ in range(self.n_goals)
                ])
                for _ in range(self.world.batch_dim)
            ]
        )

        return agent_payloads, goal_payloads

    def random_paint_generator(self, device: torch.device):
        if self.task_complexity == 'nav':
            if self.n_goals <= self.n_agents:
                goal_payloads, agent_payloads = self.premix_paints(self.n_goals, self.n_agents, device)
            else:
                agent_payloads, goal_payloads = self.premix_paints(self.n_agents, self.n_goals, device)

            # test = goal_payloads.unsqueeze(0).expand(self.world.batch_dim, self.n_goals, 3)
            # goal_payloads = goal_payloads.unsqueeze(0).repeat(self.world.batch_dim, 1)
        else:
            agent_payloads, goal_payloads = self.unmixed_paints(device)
        # TODO: Full problem: create a set of n_goals from a mix of RGB agent payloads.
        return agent_payloads, goal_payloads

    def reset_agents(self, env_index):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.goals,
            self.world,
            env_index,
            min_dist_between_entities=1,
            x_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2)),
            y_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2))
        )

        agent_payload, goal_payload = self.random_paint_generator(self.world.device)

        for i, agent in enumerate(self.world.agents):
            # Form a payload of shape [Batch_dim, RGB source, RGB mixed] default mixed to source value.
            payload = agent_payload[:, i, :].unsqueeze(1).repeat(1, 2, 1)
            agent.set_payload(payload, batch_index=env_index)

        for i, goal in enumerate(self.goals):
            # Goal should be of shape: [Batch dim, RGB]
            goal.set_expected_payload(goal_payload[:, i, :], batch_index=env_index)

        # Preliminary position shaping to all goals. Clone distances to use as normalisation during reward computation.
        for agent in self.world.agents:
            if env_index is None:
                agent.shaping = (
                        torch.stack(
                            [torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=1) for goal in
                             self.goals])
                        * self.pos_shaping_factor
                )
                agent.pos_shape_norm = agent.shaping.clone()
            else:
                agent.shaping[env_index] = (
                        torch.stack(
                            [torch.linalg.vector_norm((agent.state.pos[env_index] - goal.state.pos[env_index]), dim=1)
                             for goal in
                             self.goals])
                        * self.pos_shaping_factor
                )
                agent.pos_shape_norm = agent.shaping.clone()

    def reset_world_at(self, env_index: int = None):
        self.reset_agents(env_index)
        self.world.reset_map(env_index)

    def observation(self, agent: DOTSAgent) -> AGENT_OBS_TYPE:
        # TODO: Test if adding this reduces collisions/improves training.. first try suggests not.
        # Get vector norm distance to all other agents.
        other_agents = torch.transpose(
            torch.stack(
                [
                    torch.linalg.vector_norm((agent.state.pos - a.state.pos), dim=1)
                    for a in self.world.agents if a != agent
                ]
            ), 0, 1
        ) if self.observe_other_agents else torch.empty(0)

        # Forms a list of tensors [distX, distY, R, G, B] for each goal. TODO: Seems too hard to learn..?
        #   Shape [Batch size, n_goals * (2 + payload_dims)]
        if self.observe_all_goals:
            goals = [
                torch.cat(
                    [
                        goal.state.pos - agent.state.pos,
                        goal.state.expected_payload
                    ],
                    dim=-1)
                for goal in self.goals
            ]

        else:
            mixed_payload = agent.state.payload[:, 1, :]
            # Restrict observations to just the correctly coloured goal.
            goals = [
                torch.cat(
                    [
                        goal.state.pos - agent.state.pos
                    ],
                    dim=-1)
                for goal in self.goals if torch.any(torch.eq(goal.state.expected_payload, mixed_payload))
            ]

        return torch.cat(
            ([
                 agent.state.pos,
                 agent.state.vel,
                 torch.reshape(agent.state.payload, (self.world.batch_dim, -1))
             ]
             + [*goals]
             + [other_agents]),
            dim=-1,
        )

    # Individual agent reward structure.
    def reward(self, agent: DOTSAgent) -> AGENT_REWARD_TYPE:
        agent.agent_collision_rew[:] = 0
        agent.obstacle_collision_rew[:] = 0
        agent.agent_pos_reward[:] = 0
        agent.agent_final_reward[:] = 0

        # Collects a tensor of goals with the same colour as the mixed payload.
        color_match = torch.stack(
            [(torch.floor(
                torch.sum(torch.eq(agent.state.payload[:, 1, :], goal.state.expected_payload),
                          dim=-1) / 3))
                for goal in self.goals])

        # Collects distances to all goal boundaries. Currently assumes square goals
        dists = torch.stack(
            [
                torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=1)
                - goal.shape.length / 2
                for goal in self.goals
            ]
        )

        if self.pos_shaping:
            # Perform position shaping on goal distances
            pos_shaping = dists * self.pos_shaping_factor
            shaped_dists = (agent.shaping - pos_shaping) / agent.pos_shape_norm
            agent.shaping = pos_shaping

            # Calculate positional reward, and update global reward info
            agent.agent_pos_reward += (shaped_dists * color_match).sum(dim=0)
            self.pos_rew += agent.agent_pos_reward

        # Return true if agent with correct payload is on goal with same expected payload.
        on_goal = (dists * color_match).sum(dim=0) < agent.shape.radius / 2

        # Save per agent final reward iff on_goal.
        agent.agent_final_reward[on_goal] += self.final_reward / self.n_agents

        if self.agent_collision_penalty != 0:
            for a in self.world.agents:
                if a != agent:
                    agent.agent_collision_rew[
                        self.world.get_distance(agent, a) <= self.min_collision_distance
                        ] += self.agent_collision_penalty
        if self.wall_collision_penalty != 0:
            for l in self.world.landmarks:
                if self.world.collides(agent, l):
                    if l in self.world.walls:
                        penalty = self.wall_collision_penalty
                    else:
                        penalty = 0
                    agent.obstacle_collision_rew[
                        self.world.get_distance(agent, l) <= self.min_collision_distance
                        ] += penalty

        # Only return final rewards if all agents are on goal...
        if agent == self.world.agents[-1]:
            self.final_rew[:] = 0
            for a in self.world.agents:
                self.final_rew += a.agent_final_reward
            self.final_rew[self.final_rew < self.final_reward] = 0

        return (
                agent.agent_pos_reward
                + agent.obstacle_collision_rew
                + agent.agent_collision_rew
                + self.final_rew
        )

    def done(self):
        # TODO: This is placeholder.. Return a task completion signal.
        return torch.full(
            (self.world.batch_dim,), False, device=self.world.device
        )

    def info(self, agent: DOTSAgent) -> dict:
        # TODO: Return a dictionary of reward signals to provide debugging/logging info..
        #  At the moment this is only storing the last agents calculations.
        return {"reward": self.pos_rew + self.final_rew}

    def process_action(self, agent: DOTSAgent):
        # TODO: Implement action processing logic to happen before simulation step.
        #  Utilise agent.action where u = physical action and c = comms action.
        pass

    def top_layer_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        geoms = []
        for landmark in self.world.landmarks:
            if isinstance(landmark, DOTSPayloadDest):
                l, r, t, b = 0, landmark.shape.width / 2, landmark.shape.length / 4, -landmark.shape.length / 4
                payload = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])

                col = landmark.state.expected_payload[env_index].reshape(-1)
                payload.set_color(*col)

                xform = rendering.Transform()
                payload.add_attr(xform)

                xform.set_translation(landmark.state.pos[env_index][X] - r / 2, landmark.state.pos[env_index][Y])
                geoms.append(payload)

        for agent in self.world.agents:
            primary_payload = rendering.make_circle(proportion=0.5, radius=self.agent_radius / 2)
            mixed_payload = rendering.make_circle(proportion=0.5, radius=self.agent_radius / 2)
            primary_col = agent.state.payload[env_index][0].reshape(-1)
            mixed_col = agent.state.payload[env_index][1].reshape(-1)

            primary_payload.set_color(*primary_col)
            mixed_payload.set_color(*mixed_col)

            p_xform = rendering.Transform()
            primary_payload.add_attr(p_xform)

            m_xform = rendering.Transform()
            mixed_payload.add_attr(m_xform)

            p_xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
            p_xform.set_rotation(-math.pi / 2)
            m_xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
            m_xform.set_rotation(math.pi / 2)

            geoms.append(primary_payload)
            geoms.append(mixed_payload)

            # TODO: Add agent labels?

        return geoms


if __name__ == '__main__':
    render_interactively(
        __file__,
        n_agents=3,
        n_goals=3,
        pos_shaping=True,
        communication_size=2
    )
