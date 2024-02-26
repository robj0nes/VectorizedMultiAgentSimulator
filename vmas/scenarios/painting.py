import random

import torch
import seaborn as sns

from vmas import render_interactively
from vmas.simulator.core import Landmark, Sphere, World, Box
from vmas.simulator.dots_core import DOTSWorld, DOTSAgent
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_REWARD_TYPE, AGENT_OBS_TYPE, ScenarioUtils


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

        agent_cols, goal_cols = self.random_paint_generator()

        for i in range(self.n_agents):
            agent = DOTSAgent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
                color=agent_cols[i]
            )
            # Question: Is this the best place to add this.. ?
            agent.payload = torch.tensor(agent.color, device=device).expand(batch_dim, -1)
            world.add_agent(agent)

        self.goals = []
        for i in range(self.n_goals):
            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                shape=Box(length=self.agent_radius * 2, width=self.agent_radius * 2),
                # For debugging setting goal0 == agent0 payload
                color=goal_cols[i] if i != 0 else world.agents[i].color
            )
            self.goals.append(goal)
            world.add_landmark(goal)

        # Flatten colours into shape [Batch dim, n_goals * RGB] for later observations.
        self.goal_col_obs = torch.tensor(list(sum(goal_cols, ())), device=device).expand(batch_dim, -1)

        world.spawn_map()

        # # TODO: Reward for position relative to goal iff. correct color.
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        return world

    def random_paint_generator(self):
        agent_cols = self.get_distinct_color_list()
        # TODO: Remove eventually - For trivial test problem: setting goals equal to random agent payload (no paint
        #  mixing required)
        ran_ints = random.sample(range(len(agent_cols)), self.n_goals)
        goal_cols = [agent_cols[i] for i in ran_ints]

        # # TODO: create a set of n_goals from a mix of RGB agent payloads.
        # ran_agent_cols = agent_cols.copy()
        # random.shuffle(ran_agent_cols)
        # goal_cols = [tuple(((torch.tensor(x) + torch.tensor(y)) / 2).tolist()) for (x, y) in
        #              zip(ran_agent_cols[:self.n_goals], ran_agent_cols[self.n_goals:])]
        return agent_cols, goal_cols

    def get_distinct_color_list(self):
        opts = sns.color_palette(palette="Set2")
        return [opts[x] for x in random.sample(range(0, len(opts)), self.n_agents)]

    def reset_agents(self, env_index):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.goals,
            self.world,
            env_index,
            min_dist_between_entities=1,
            x_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2)),
            y_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2))
        )

        agent_cols, goal_cols = self.random_paint_generator()

        for i, agent in enumerate(self.world.agents):
            agent.color = agent_cols[i]
            agent.payload = (torch.tensor(agent.color, device=self.world.device)
                             .expand(self.world.batch_dim, -1))

        for i, goal in enumerate(self.goals):
            goal.color = goal_cols[i]

        # Reset observable goal colours
        self.goal_col_obs = (torch.tensor(list(sum(goal_cols, ())), device=self.world.device)
                             .expand(self.world.batch_dim, -1))

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
                agent.payload,
                goal_dists,
                self.goal_col_obs
            ],
            dim=-1,
        )

    def reward(self, agent: DOTSAgent) -> AGENT_REWARD_TYPE:
        # Question: Do I need to assign these to device?
        # We find any goal color which matches the current payload color and reward based on distance from goal
        color_match = torch.stack(
            # We want floor(sum(match) / 3) as we are looking for a complete RGB match
            [(torch.floor(torch.sum(torch.eq(agent.payload, torch.tensor(goal.color)), dim=-1) / 3))
             for goal in self.goals]).to(self.world.device)

        dists = torch.stack(
            [torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=-1) for goal in self.goals]).to(
            self.world.device)

        # TODO: Temporary to establish a min radius
        # Question: There must be something like this here already.. ?
        min_dist = 2
        dists[dists >= min_dist] = float('inf')

        # TODO: Need a negative reward for positioning over incorrect goal?

        self.rew = torch.sum(color_match * 1 / dists, dim=0)
        self.rew[self.rew == float('inf')] = 0
        return self.rew

    def done(self):
        # TODO: This is placeholder.. Return a task completion signal.
        return torch.full(
            (self.world.batch_dim,), False, device=self.world.device
        )

    def info(self, agent: DOTSAgent) -> dict:
        # TODO: Return a dictionary of reward signals to provide debugging/logging info.
        return {"reward": self.rew}


if __name__ == '__main__':
    render_interactively(
        __file__,
        n_agents=4,
        n_goals=4
    )
