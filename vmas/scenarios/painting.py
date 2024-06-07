import math
import random
from enum import Enum

import numpy as np
import torch
import seaborn as sns

from vmas import render_interactively
from vmas.simulator import rendering
from vmas.simulator.core import Sphere, World, Box
from vmas.simulator.dots_core import DOTSWorld, DOTSAgent, DOTSPayloadDest
from vmas.simulator.rendering import TextLine
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_REWARD_TYPE, AGENT_OBS_TYPE, ScenarioUtils, Y, X, Color


# TODO:
#  1. Look at a navigation implementation driven by coms only.
#  2. Try to integrate coms and nav once above is solved.
#  3. Later: Implementation assumes agent[i] and goal[i] are always connected. Does not necessarily generalise

def get_distinct_color_list(n_cols, device: torch.device, exclusions=None):
    opts = sns.color_palette(palette="Set2")
    return torch.stack([torch.tensor(opts[x], device=device, dtype=torch.float32) for x in
                        random.sample(range(0, len(opts)), n_cols)])


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        # World properties
        self.arena_size = None
        self.n_goals = None
        self.n_agents = None
        self.task_type = None
        self.goals = None

        # Agent properties
        self.payload_shape = None
        self.agent_radius = None
        self.dim_c = None
        self.agent_action_size = None

        # Observation and reward properties
        self.observe_other_agents = None
        self.observe_all_goals = None
        self.integrated_coms = None
        self.coms_proximity = None
        self.observation_proximity = None
        self.mixing_thresh = 0.01
        self.learn_mix = None
        self.learn_coms = None
        self.rew = None

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = kwargs.get("n_agents", 4)
        self.agent_radius = 0.2
        self.arena_size = 5
        self.viewer_zoom = 1.7

        # Payload is of shape: [2 (source, mix), payload dim (eg. 3-RGB)]
        self.payload_shape = kwargs.get("payload_shape", (2, 3))

        self.n_goals = kwargs.get("n_goals", 4)
        self.observation_proximity = kwargs.get("observation_proximity", self.arena_size)
        self.observe_all_goals = kwargs.get("observe_all_goals", False)
        self.observe_other_agents = kwargs.get("observe_other_agents", True)
        self.integrated_coms = kwargs.get("integrated_coms", True)
        self.coms_proximity = kwargs.get("coms_proximity", self.arena_size)
        self.mixing_thresh = kwargs.get("mixing_thresh", 0.01)
        self.learn_mix = kwargs.get("learn_mix", True)
        self.learn_coms = kwargs.get("learn_coms", True)

        self.task_type = kwargs.get("task_type", "nav")

        # Size of coms = No coms if only testing navigation,
        #   else: 1-D mixing intent + N-D payload.
        self.dim_c = kwargs.get("dim_c", 1 + self.payload_shape[1]) \
            if self.task_type != 'nav' else 0

        # Size of action = 2-D force + N-D payload co-efficients.
        self.agent_action_size = kwargs.get("action_size", 2 + self.payload_shape[1])

        # dim_c is the size of the communications signal.
        world = DOTSWorld(batch_dim, device, collision_force=100, dim_c=self.dim_c)

        for i in range(self.n_agents):
            agent = DOTSAgent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                color=Color.GREEN,
                payload_shape=self.payload_shape,
                silent=True if self.dim_c == 0 else False,
                # TODO: Make action size > 2 to learn mixing weights.
                action_size=self.agent_action_size
            )
            world.add_agent(agent)
            agent.agent_collision_rew = torch.zeros(batch_dim, device=device)
            agent.obstacle_collision_rew = agent.agent_collision_rew.clone()
            agent.agent_pos_reward = agent.agent_collision_rew.clone()
            agent.agent_final_reward = agent.agent_collision_rew.clone()
            agent.agent_mixing_reward = agent.agent_collision_rew.clone()

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
        self.env_collision_penalty = kwargs.get("env_collision_penalty", -0.2)
        self.min_collision_distance = kwargs.get("collision_dist", 0.005)

        self.pos_shaping = kwargs.get("pos_shaping", False)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1.0)

        self.mix_shaping = kwargs.get("mix_shaping", False)
        self.mix_shaping_factor = kwargs.get("mix_shaping_factor", 1.0)

        self.all_on_goal = kwargs.get("final_pos_reward", 0.05)
        self.all_mixed = kwargs.get("final_mix_reward", 0.05)
        self.per_agent_reward = kwargs.get("per_agent_reward", False)

        self.final_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.final_pos_rew = self.final_rew.clone()
        self.final_mix_rew = self.final_rew.clone()
        return world

    def premix_paints(self, small, large, device):
        large_payloads = torch.stack(
            [
                get_distinct_color_list(large, device) for _ in range(self.world.batch_dim)
            ]
        )

        # ran_ints = [
        #     random.sample(range(large_payloads.shape[1]), small) for _ in range(self.world.batch_dim)
        # ]

        # TODO: Test nav with random agent-goal assignments. Make sure that observations are
        #  based on colour match, not agent index.
        small_payloads = torch.stack(
            [
                torch.stack(
                    [
                        # Random goal assignment
                        # large_payloads[j][i] for i in ran_ints[j]

                        # Indexed goal assignment
                        large_payloads[j][i] for i in range(large_payloads.shape[1])
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

    # TODO: Have some check to ensure all goal mixes are unique.
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
                        ).astype(np.float32)
                    ),
                    device=device)
                for _ in range(self.world.batch_dim)
            ]
        )
        agent_payloads /= 255

        # Generate a random colour in RGB colour space for the goals.
        goal_payloads = torch.stack(
            [
                torch.stack([
                    torch.tensor(np.random.uniform(0.01, 1.0, 3), device=device, dtype=torch.float32)
                    for _ in range(self.n_goals)
                ])
                for _ in range(self.world.batch_dim)
            ]
        )

        return agent_payloads, goal_payloads

    def random_paint_generator(self, device: torch.device):
        if self.task_type == 'nav':
            if self.n_goals <= self.n_agents:
                goal_payloads, agent_payloads = self.premix_paints(self.n_goals, self.n_agents, device)
            else:
                agent_payloads, goal_payloads = self.premix_paints(self.n_agents, self.n_goals, device)
        else:
            # TODO: Full problem: create a set of n_goals from a mix of RGB agent payloads.
            agent_payloads, goal_payloads = self.unmixed_paints(device)
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

        # Preliminary position and mix shaping to all goals.
        # Clone distances to use as normalisation during reward computation.
        for agent in self.world.agents:
            if env_index is None:
                agent.shaping = (
                        torch.stack(
                            [torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=1) for goal in
                             self.goals])
                        * self.pos_shaping_factor)

                agent.mix_shaping = (
                        torch.stack(
                            [
                                torch.linalg.vector_norm((agent.state.payload[:, 1, :] - goal.state.expected_payload),
                                                         dim=1)
                                for goal in self.goals
                            ])
                        * self.mix_shaping_factor)
            else:
                agent.shaping[env_index] = (
                        torch.stack(
                            [torch.linalg.vector_norm((agent.state.pos[env_index] - goal.state.pos[env_index]), dim=1)
                             for goal in self.goals])
                        * self.pos_shaping_factor
                )
                agent.mix_shaping[env_index] = (
                        torch.stack(
                            [
                                torch.linalg.vector_norm(
                                    (agent.state.payload[env_index, 1, :] - goal.state.expected_payload[env_index]),
                                    dim=1)
                                for goal in self.goals
                            ])
                        * self.mix_shaping_factor)
            agent.mix_shaping_norm = agent.mix_shaping.clone()
            agent.pos_shape_norm = agent.shaping.clone()

    def reset_world_at(self, env_index: int = None):
        self.reset_agents(env_index)
        self.world.reset_map(env_index)

    # TODO:
    #  1. Implement a restriction on observations based on the self.observation_proximity value.
    #       - Maybe look at how lidar is implemented for hints on variable observation space..
    #  2. Work on communications observation once coms GNN is implemented

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

        task_obs = [self.compute_goal_observations(agent)]
        # if self.task_type != "mix":
        #     # Collect distance deltas to goals.
        #     task_obs.append(self.compute_goal_observations(agent))

        if self.task_type != "nav":
            if self.integrated_coms:
                if self.learn_coms:
                    # Collect communications from other agents.
                    agent_coms = torch.stack(
                        [a.state.c for a in self.world.agents if a != agent]
                    )
                else:
                    # Assume agents communicate correctly:
                    #   Directly observe the source payload of the other agents.
                    agent_coms = torch.stack(
                        [a.state.payload[:, 0, :] for a in self.world.agents if a != agent]
                    )

                for c in agent_coms:
                    task_obs.append(c)
            else:
                # TODO: Get coms embedding from corresponding agent node in external coms GNN
                pass

        return torch.cat(
            ([
                 agent.state.pos,
                 agent.state.vel,
                 agent.state.payload[:, 0, :],
                 agent.state.payload[:, 1, :]
             ]
             + [*task_obs]
             + [other_agents]),
            dim=-1,
        )

    # TODO: The full implementation for this will require some work as we need to define some variable
    #  observation of whether a payload-goal colour match has been made in a given batch dim.
    #   - Perhaps best to always observe goal, but only receive positional rewards once colour match has been made.
    #  ..
    def compute_goal_observations(self, agent):
        # Forms a list of tensors [distX, distY, R, G, B] for each goal.
        #   Shape [Batch size, n_goals * (2 + payload_dims)]
        if self.observe_all_goals:
            # If we observe all goals, simply compute distance to each goal.
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
            # TODO: Work out how to filter goals by colour match. Observe only goals with correct match.
            agent_index = int(agent.name.split('_')[1])

            if self.task_type == "mix":
                # Goal obs is just the distance from expected payload.
                # goals = self.goals[agent_index].state.expected_payload
                goals = torch.cat(
                    [self.goals[agent_index].state.expected_payload - agent.state.payload[:, 1, :]]
                )
            else:
                # Goal obs is relative position and payload.
                goals = torch.cat(
                    [
                        self.goals[agent_index].state.pos - agent.state.pos,
                        self.goals[agent_index].state.expected_payload
                    ],
                    dim=-1)

        return goals

    # Individual agent reward structure.
    def reward(self, agent: DOTSAgent) -> AGENT_REWARD_TYPE:
        agent.agent_collision_rew[:] = 0
        agent.obstacle_collision_rew[:] = 0
        agent.agent_mixing_reward[:] = 0
        agent.agent_pos_reward[:] = 0
        agent.agent_final_reward[:] = 0

        self.compute_collision_penalties(agent)

        # If we are the last agent, who computes global reward signals, reset final reward.
        if agent == self.world.agents[-1]:
            self.final_rew[:] = 0

        # If we are just training mixing, don't bother with navigation.
        if self.task_type != "mix":
            self.compute_positional_rewards(agent)

            if agent == self.world.agents[-1]:
                self.final_pos_rew[:] = 0
                for a in self.world.agents:
                    self.final_pos_rew += a.agent_final_reward

                if self.per_agent_reward:
                    # Final reward is proportional to num agents who have reached their goal.
                    self.final_rew += self.final_pos_rew
                else:
                    # Zero any batch dim when one or more agents have not reached their goal.
                    self.final_pos_rew[self.final_pos_rew < self.all_on_goal] = 0
                    # Add all_on_goal reward to batch dims when all agents have reached their goals.
                    self.final_rew[self.final_pos_rew > 0] += self.all_on_goal

        # If we are just training navigation, don't bother with mixing.
        if self.task_type != "nav":
            self.compute_mixing_rewards(agent)

            # If all agents have successfully mixed, add a final mixing reward.
            if agent == self.world.agents[-1]:
                self.final_mix_rew[:] = 0
                for a in self.world.agents:
                    # Seeking goal should be {0, 1} so multiply by total mixing reward / num agents
                    self.final_mix_rew += a.state.seeking_goal * (self.all_mixed / self.n_agents)

                if self.per_agent_reward:
                    # Final reward is proportional to num agents who have reached their goal.
                    self.final_rew += self.final_mix_rew
                else:
                    # Zero any batch dim when one or more agents have not mixed.
                    self.final_mix_rew[self.final_mix_rew < self.all_mixed] = 0
                    # Add all_mixed reward to batch dims when all agents have mixed the correct solution.
                    self.final_rew[self.final_mix_rew > 0] += self.all_mixed

        name = agent.name
        a_pos = agent.agent_pos_reward
        a_mix = agent.agent_mixing_reward
        a_acol = agent.agent_collision_rew
        a_ocol = agent.obstacle_collision_rew
        final = self.final_rew

        return (
                agent.agent_pos_reward
                + agent.agent_mixing_reward
                + agent.obstacle_collision_rew
                + agent.agent_collision_rew
                + self.final_rew
        )

    def compute_positional_rewards(self, agent):
        # Collects a tensor of goals with the same colour as the mixed payload.
        colour_match = torch.stack(
            [
                torch.linalg.vector_norm(
                    (agent.state.payload[:, 1, :] - goal.state.expected_payload), dim=1) < self.mixing_thresh
                for goal in self.goals
            ])

        dists = torch.stack(
            [
                torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=1)
                # - goal.shape.length / 2
                for goal in self.goals
            ])

        if self.pos_shaping:
            # Perform position shaping on goal distances
            pos_shaping = dists * self.pos_shaping_factor
            shaped_dists = (agent.shaping - pos_shaping) / agent.pos_shape_norm

            agent.shaping = pos_shaping

            # Calculate positional reward, and update global reward info
            agent.agent_pos_reward += (shaped_dists * colour_match).sum(dim=0)

        # Return true if agent with correct payload is on goal with same expected payload.
        matched_dists = torch.abs((dists * colour_match).sum(dim=0))
        # on_goal = torch.eq(0 < matched_dists, matched_dists < 2 * agent.shape.radius)
        on_goal = torch.eq(0 < matched_dists, matched_dists < agent.shape.radius / 2)

        agent.agent_final_reward[on_goal] += self.all_on_goal / self.n_agents

    def compute_mixing_rewards(self, agent):
        # TODO: Currently selects a goal based on the agent index.. not very good for generalising.
        index = int(agent.name.split('_')[1])

        payload_dists = torch.linalg.vector_norm(
            (agent.state.payload[:, 1, :] - self.goals[index].state.expected_payload), dim=1)
        # Dist below threshold
        agent.state.seeking_goal = torch.logical_or(agent.state.seeking_goal, payload_dists < self.mixing_thresh)

        if self.mix_shaping:
            # Perform reward shaping on payload mixtures.
            mix_shaping = payload_dists * self.mix_shaping_factor
            shaped_mixes = (agent.mix_shaping[index] - mix_shaping) / agent.mix_shaping_norm[index]
            agent.mix_shaping[index] = mix_shaping
            agent.agent_mixing_reward += shaped_mixes

        if self.world.batch_dim > 1:
            # Squeeze to conform to reward return structure.
            agent.agent_mixing_reward = agent.agent_mixing_reward.squeeze()

    def compute_collision_penalties(self, agent):
        if self.agent_collision_penalty != 0:
            for a in self.world.agents:
                if a != agent:
                    agent.agent_collision_rew[
                        self.world.get_distance(agent, a) <= self.min_collision_distance
                        ] += self.agent_collision_penalty
        if self.env_collision_penalty != 0:
            for l in self.world.landmarks:
                if self.world.collides(agent, l):
                    if l in self.world.walls:
                        penalty = self.env_collision_penalty
                    else:
                        penalty = 0
                    agent.obstacle_collision_rew[
                        self.world.get_distance(agent, l) <= self.min_collision_distance
                        ] += penalty

    def done(self):
        # TODO: This is placeholder.. Return a task completion signal.
        return torch.full(
            (self.world.batch_dim,), False, device=self.world.device
        )

    def info(self, agent: DOTSAgent) -> dict:
        # TODO: Return a dictionary of reward signals to provide debugging/logging info..
        #  At the moment this is only storing the last agents calculations.
        return {
            "pos_reward": agent.agent_pos_reward,
            "mix_reward": agent.agent_mixing_reward,
            "final_rew": self.final_rew
        }

    # TODO: Implement
    #  1. A speaking exercise: Learn coms, known mixing coefficients (done)
    #  2. A listening exercise: Known coms, learns mixing coefficients.
    #  3. Combined exercise: Learn coms, Learn mixing coefficients.
    def mix_payloads(self, agent: DOTSAgent):
        # TODO: Allow re-mixing after correct mix has been produced. Ie. remove constraint to ensure proper learning.
        """
        Uses the agent coms signal (c[0]) to determine if mix is desired.
        Uses the other agent coms signal (c[1:]) to read the other payload.
        Args:
            agent: The agent requesting the payload mix.
        """
        agent_index = int(agent.name.split('_')[1])

        # Establish where a request is true along the batch dim,
        #  and the agent hasn't already mixed the correct payload.
        #   Copy for (n_agents - 1) to simplify tensor calcs.
        request_mix = ((torch.logical_and(agent.action.c[:, 0] > 0.5, ~agent.state.seeking_goal.squeeze()))
                       .unsqueeze(0).repeat(self.n_agents - 1, 1))

        # Find all agents in proximity along the batch dim. Shape: [n_agents-1, batch dim]
        #  Multiply by request_mix to 0 any batch dim where the agent is not requesting to mix.
        in_prox = (((torch.stack(
            [torch.linalg.norm(agent.state.pos - other.state.pos, dim=1)
             for other in self.world.agents if other != agent]))
                    < self.coms_proximity) * request_mix)

        # Clone existing agent mixing payload.
        new_mix = agent.state.payload[:, 1, :].clone()

        # Collect agent mixing weights across batch dim.
        if self.learn_mix:
            # Mixing weights should be the last n values of the action vector, where n is the dimension of the payload.
            payload_dims = self.payload_shape[-1]
            mix_coefficients = (agent.action.u[:, -payload_dims:] + 1) / 2  # Shift from [-1, 1] to [0,1]
        else:
            mix_coefficients = self.goals[agent_index].state.expected_payload

        # Collect incoming coms signals across the batch dim.
        if self.learn_coms:
            # Get the communicated payload from other agents coms signals
            com_payloads = torch.stack(
                [other.state.c[:, 1:] for other in self.world.agents if other != agent]
            )
        else:
            # If not learning to communicate, ignore coms signals and just take collect source payloads.
            com_payloads = torch.stack(
                [other.state.payload[:, 0, :] for other in self.world.agents if other != agent]
            )

        # Zero all mixing payloads which will be updated.
        for i in range(in_prox.shape[0]):
            new_mix[in_prox[i], :] = 0

        # TODO: Remove - used for debugging coms signals.
        agent.incoming_coms = new_mix.clone()
        for i in range(in_prox.shape[0]):
            agent.incoming_coms[in_prox[i], :] += com_payloads[i, in_prox[i], :]
        comb_in_prox = torch.logical_or(*in_prox)
        agent.incoming_coms[comb_in_prox, :] += agent.state.payload[comb_in_prox, 0, :]

        # Update the agent mixing payload with the weighted mix of the others' payload.
        for i in range(in_prox.shape[0]):
            new_mix[in_prox[i], :] += com_payloads[i, in_prox[i], :] * mix_coefficients[in_prox[i], :]

        # Add agents own payload combination along any dim where a mix has been made.
        comb_in_prox = torch.logical_or(*in_prox)
        new_mix[comb_in_prox, :] += agent.state.payload[comb_in_prox, 0, :] * mix_coefficients[comb_in_prox, :]

        agent.state.payload[:, 1, :] = new_mix

    def process_action(self, agent: DOTSAgent):
        if self.task_type != "nav":
            self.mix_payloads(agent)

    def top_layer_render(self, env_index: int = 0):
        geoms = []

        # Render landmark payloads
        for landmark in self.world.landmarks:
            if isinstance(landmark, DOTSPayloadDest):
                l, r, t, b = 0, landmark.shape.width / 2, landmark.shape.length / 4, -landmark.shape.length / 4
                payload = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])

                col = landmark.state.expected_payload[env_index].reshape(-1)
                payload.set_color(*col)

                xform = rendering.Transform()
                xform.set_translation(landmark.state.pos[env_index][X] - r / 2, landmark.state.pos[env_index][Y])
                payload.add_attr(xform)
                geoms.append(payload)

        # Re-render agents (on top of landmarks) and agent payloads.
        for i, agent in enumerate(self.world.agents):
            # test = agent.render(env_index=env_index)
            for elem in agent.render(env_index=env_index):
                geoms.append(elem)
            primary_payload = rendering.make_circle(proportion=0.5, radius=self.agent_radius / 2)
            mixed_payload = rendering.make_circle(proportion=0.5, radius=self.agent_radius / 2)
            primary_col = agent.state.payload[env_index][0].reshape(-1)
            mixed_col = agent.state.payload[env_index][1].reshape(-1)

            # Add a yellow ring around agents who have successfully matched their payloads.
            if agent.state.seeking_goal[env_index]:
                success_ring = rendering.make_circle(radius=self.agent_radius, filled=True)
                success_ring.set_color(1, 1, 0)
                s_xform = rendering.Transform()
                s_xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
                success_ring.add_attr(s_xform)
                geoms.append(success_ring)

            primary_payload.set_color(*primary_col)
            mixed_payload.set_color(*mixed_col)

            p_xform = rendering.Transform()
            primary_payload.add_attr(p_xform)

            m_xform = rendering.Transform()
            mixed_payload.add_attr(m_xform)

            p_xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
            p_xform.set_rotation(math.pi / 2)
            m_xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
            m_xform.set_rotation(-math.pi / 2)

            # label = TextLine(f"Agent {i}", x=agent.state.pos[env_index][X], y=agent.state.pos[env_index][Y] - 10)
            # geoms.append(label)

            geoms.append(primary_payload)
            geoms.append(mixed_payload)

            # TODO: Using for debugging, one or other to avoid overlapping text.
            self.render_rewards(agent, env_index, geoms)
            self.render_actions(agent, env_index, geoms)

            # TODO: Add agent labels?

        return geoms

    def render_actions(self, agent, env_index, geoms):
        if agent.action.u is not None:
            agent_index = int(agent.name.split('_')[1])

            incoming_coms = ("coms_in: ["
                             + ",".join([f"{c:.2f}" for c in agent.incoming_coms[env_index]])
                             + "]")
            coms_line = rendering.TextLine(f'{agent.name} {incoming_coms}',
                                           y=(490 - (35 * (int(agent.name.split('_')[-1]) + 1))))

            action = ("vel: ["
                      + ",".join([f"{a:.2f}" for a in agent.action.u[env_index][:-self.payload_shape[-1]]])
                      + "] mix_weights: ["
                      + ",".join([f"{(a + 1) / 2:.2f}" for a in agent.action.u[env_index][-self.payload_shape[-1]:]])
                      + "]")
            action_line = rendering.TextLine(f'{agent.name} {action}',
                                             y=(350 - (35 * (int(agent.name.split('_')[-1]) + 1))))

            payload = ("payload: ["
                       + ",".join([f"{p:.2f}" for p in agent.state.payload[env_index, 1, :]])
                       + "]  goal: ["
                       + ",".join([f"{g:.2f}" for g in self.goals[agent_index].state.expected_payload[env_index]])
                       + "]"
                       )
            payload_line = rendering.TextLine(f'{agent.name} {payload}',
                                              y=(230 - (35 * (int(agent.name.split('_')[-1]) + 1))))
            geoms.append(coms_line)
            geoms.append(action_line)
            geoms.append(payload_line)

    def render_rewards(self, agent, env_index, geoms):
        rew = (agent.agent_collision_rew +
               agent.agent_mixing_reward +
               agent.agent_pos_reward +
               agent.obstacle_collision_rew +
               agent.agent_final_reward)

        reward = rendering.TextLine(f'{agent.name} rew: {rew[env_index]:.2f}', x=0,
                                    y=(650 - (35 * (int(agent.name.split('_')[-1]) + 1))))
        geoms.append(reward)

        if agent == self.world.agents[-1]:
            final_rew = rendering.TextLine(f'final rew: {self.final_rew[env_index]:.2f}', x=0, y=650)
            geoms.append(final_rew)


if __name__ == '__main__':
    render_interactively(
        __file__,
        n_agents=3,
        n_goals=3,
        pos_shaping=True,
        mix_shaping=True,
        task_type="nav",
        action_size=2,
        clamp_actions=True,
        agent_collision_penalty=-0.2,
        env_collision_penalty=-0.2
    )
