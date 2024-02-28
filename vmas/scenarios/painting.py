import random
from typing import List

import torch
import seaborn as sns

from vmas import render_interactively
from vmas.simulator.core import Landmark, Sphere, World, Box
from vmas.simulator.dots_core import DOTSWorld, DOTSAgent
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_REWARD_TYPE, AGENT_OBS_TYPE, ScenarioUtils, Y, X, Color


# Question:
#   1. How can I define agent action space
#   2. Advice on adding batched properties properly (eg. agent.colour, goal.colour, payload)
#   3. Rewards: Advice on how to form well structured reward signals...
#   4. Extending the evaluation video length.

class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.rew = None
        self.goal_col_obs = None
        self.arena_size = None
        self.n_goals = None
        self.agent_radius = None
        self.n_agents = None
        self.goals = None

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = kwargs.get("n_agents", 4)
        self.agent_radius = 0.2

        self.n_goals = kwargs.get("n_goals", 2)

        self.arena_size = 5
        self.viewer_zoom = 1.7

        world = DOTSWorld(batch_dim, device, collision_force=400, substeps=5)

        agent_payload, goal_cols = self.random_paint_generator(device)

        for i in range(self.n_agents):
            agent = DOTSAgent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
                color=Color.GREEN,
                payload_shape=3
            )
            # Question: I want to set unique agent colours along the batch dimension (equal to payload)
            world.add_agent(agent)
            agent.agent_collision_rew = torch.zeros(batch_dim, device=device)
            agent.obstacle_collision_rew = agent.agent_collision_rew.clone()

        self.goals = []
        for i in range(self.n_goals):
            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                shape=Box(length=self.agent_radius * 2, width=self.agent_radius * 2),
                # For debugging setting goal0 == agent0 payload
                color=goal_cols[i]
            )
            self.goals.append(goal)
            # Question: Similar to agents, I want to set goal colours along the batch dimension so they are unique in
            #  each env instance.
            world.add_landmark(goal)

        # Expand goal cols into shape [Batch dim, n_goals * RGB] for later observations.
        self.goal_col_obs = goal_cols.reshape(-1).unsqueeze(0).repeat(batch_dim, 1)

        world.spawn_map()

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", 0)
        self.wall_collision_penalty = kwargs.get("passage_collision_penalty", 0)
        self.pos_shaping_factor = kwargs.get("position_shaping_factor", 0)
        self.min_collision_distance = 0.005

        # # TODO: Reward for position relative to goal iff. correct color.
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.final_rew = self.pos_rew.clone()
        return world

    def random_paint_generator(self, device: torch.device):
        agent_cols = self.get_distinct_color_list(device)
        # TODO: Remove eventually - For trivial test problem: setting goals equal to random agent payload (no paint
        #  mixing required)
        ran_ints = random.sample(range(len(agent_cols)), self.n_goals)
        goal_cols = torch.stack([agent_cols[i] for i in ran_ints])

        # # TODO: create a set of n_goals from a mix of RGB agent payloads.
        # ran_agent_cols = agent_cols.copy()
        # random.shuffle(ran_agent_cols)
        # goal_cols = [tuple(((torch.tensor(x) + torch.tensor(y)) / 2).tolist()) for (x, y) in
        #              zip(ran_agent_cols[:self.n_goals], ran_agent_cols[self.n_goals:])]
        return agent_cols, goal_cols

    def get_distinct_color_list(self, device: torch.device):
        opts = sns.color_palette(palette="Set2")
        return torch.stack([torch.tensor(opts[x], device=device, dtype=torch.float32) for x in
                            random.sample(range(0, len(opts)), self.n_agents)])

    def reset_agents(self, env_index):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.goals,
            self.world,
            env_index,
            min_dist_between_entities=1,
            x_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2)),
            y_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2))
        )

        agent_payload, goal_cols = self.random_paint_generator(self.world.device)

        # Question can we set agent and goal colors across the batches..?
        for i, agent in enumerate(self.world.agents):
            # agent.color = agent_payload[i]
            payload = agent_payload[i].unsqueeze(0).repeat(self.world.batch_dim, 1)
            agent.set_payload(payload, batch_index=env_index)

        for i, goal in enumerate(self.goals):
            goal.color = goal_cols[i]

        # Reset observable goal colours
        # Expand goal cols into shape [Batch dim, n_goals * RGB] for later observations.
        self.goal_col_obs = goal_cols.reshape(-1).unsqueeze(0).repeat(self.world.batch_dim, 1)

        # Preliminary position shaping to all goals.
        for agent in self.world.agents:
            if env_index is None:
                agent.shaping = (
                        torch.stack(
                            [torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=-1) for goal in
                             self.goals])
                        * self.pos_shaping_factor
                )
            else:
                agent.shaping[env_index] = (
                        torch.stack(
                            [torch.linalg.vector_norm((agent.state.pos[env_index] - goal.state.pos[env_index]), dim=-1)
                             for goal in
                             self.goals])
                        * self.pos_shaping_factor
                )

    def reset_world_at(self, env_index: int = None):
        self.reset_agents(env_index)
        self.world.reset_map(env_index)

    def observation(self, agent: DOTSAgent) -> AGENT_OBS_TYPE:
        # Question: Should these be assigned to device?
        # Shape = [Batch size, n_goals * 2] (Position of goals (x,y concatenated))
        goal_dists = torch.cat([goal.state.pos - agent.state.pos for goal in self.goals], dim=-1)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.payload,
                goal_dists,
                self.goal_col_obs
            ],
            dim=-1,
        )

    def reward(self, agent: DOTSAgent) -> AGENT_REWARD_TYPE:
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for agent in self.world.agents:
                # Returns a tensor of goals with the same colour as the payload.
                color_match = torch.stack(
                    [(torch.floor(
                        torch.sum(torch.eq(agent.state.payload, goal.color.clone().detach().requires_grad_(True)),
                                  dim=-1) / 3))
                        for goal in self.goals])

                # Returns distances to all goals.
                dists = torch.stack(
                    [torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=-1) for goal in self.goals])

                # Perform position shaping on goal distances
                pos_shaping = dists * self.pos_shaping_factor
                shaped_dists = agent.shaping - pos_shaping
                agent.shaping = pos_shaping

                self.pos_rew += (shaped_dists * color_match).sum(dim=0)

                # TODO: Calculate a final reward if agent is on goal and correct colour.

            agent.agent_collision_rew[:] = 0
            agent.obstacle_collision_rew[:] = 0
            for a in self.world.agents:
                if a != agent:
                    agent.agent_collision_rew[
                        self.world.get_distance(agent, a) <= self.min_collision_distance
                        ] += self.agent_collision_penalty
            for l in self.world.landmarks:
                if self.world.collides(agent, l):
                    if l in self.world.walls:
                        penalty = self.wall_collision_penalty
                    else:
                        penalty = 0
                    agent.obstacle_collision_rew[
                        self.world.get_distance(agent, l) <= self.min_collision_distance
                        ] += penalty

        return (
                self.pos_rew
                + self.final_rew
                + agent.obstacle_collision_rew
                + agent.agent_collision_rew
        )

    def done(self):
        # TODO: This is placeholder.. Return a task completion signal.
        return torch.full(
            (self.world.batch_dim,), False, device=self.world.device
        )

    def info(self, agent: DOTSAgent) -> dict:
        # TODO: Return a dictionary of reward signals to provide debugging/logging info.
        return {"reward": self.rew}

    def top_layer_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        geoms = []
        for agent in self.world.agents:
            payload = rendering.make_circle(radius=self.agent_radius / 2)

            col = agent.state.payload[env_index].reshape(-1)
            payload.set_color(*col)

            xform = rendering.Transform()
            payload.add_attr(xform)

            xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
            geoms.append(payload)

            # TODO: Add agent labels?
        return geoms


if __name__ == '__main__':
    render_interactively(
        __file__,
        n_agents=4,
        n_goals=4
    )
