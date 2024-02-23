import math
import torch

from vmas import render_interactively
from vmas.simulator.core import Landmark, Sphere, World, Box
from vmas.simulator.dots_core import DOTSWorld, DOTSAgent
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, AGENT_REWARD_TYPE, AGENT_OBS_TYPE, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = kwargs.get("n_agents", 4)
        self.agent_radius = 0.2
        assert self.n_agents > 1
        assert self.n_agents % 2 == 0, f"This scenario requires an even number of agents."

        self.arena_size = 5
        self.viewer_zoom = 1.7

        self.n_collection_points = kwargs.get("n_collection_points", 1)
        self.col_point_size = self.agent_radius * 2.2
        assert self.n_collection_points >= 1

        self.n_blueprints = kwargs.get("n_blueprints", math.floor(self.n_agents / 2))
        assert self.n_blueprints > 0

        # Make a default world
        world = DOTSWorld(batch_dim, device, collision_force=400, substeps=5)

        # Instantiate agents
        # TODO: Update agent action options to include:
        #   1. Collect material
        #   2. Connect materials
        #   3. Place blueprint
        for i in range(self.n_agents):
            agent = DOTSAgent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
                color=Color.GREEN
            )
            world.add_agent(agent)

        # Instantiate collection points
        self.col_points = []
        for i in range(self.n_collection_points):
            col_point = Landmark(
                name=f"collection_point_{i}",
                collide=False,
                shape=Box(length=self.col_point_size, width=self.col_point_size),
                color=Color.RED
            )
            self.col_points.append(col_point)
            world.add_landmark(col_point)

        self.blueprints = []
        for i in range(self.n_blueprints):
            blueprint = Landmark(
                name="blueprint",
                collide=False,
                shape=Box(length=self.col_point_size, width=self.col_point_size * 2),
                color=Color.BLUE
            )
            self.blueprints.append(blueprint)
            world.add_landmark(blueprint)

        world.spawn_map()

        # TODO: Replace with actual reward signals.
        #   Positive rewards:
        #       1. Collecting material from collection point
        #       2. Connecting material with correct match
        #       3. Placing material at blueprint
        #   Negative rewards:
        #       1. Placing material at blueprint if not connected to matching part.
        #       2. Connecting incorrect parts.
        #       3. Going to blueprint without part?
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()

        return world

    def reset_agents(self, env_index):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.col_points + self.blueprints,
            self.world,
            env_index,
            min_dist_between_entities=1,
            x_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2)),
            y_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2))
        )

    def reset_world_at(self, env_index: int = None):
        self.world.reset_map(env_index)
        self.reset_agents(env_index)

    def observation(self, agent: DOTSAgent) -> AGENT_OBS_TYPE:
        # TODO: Return all observables within agents FOV
        col_point_dists = torch.cat([point.state.pos - agent.state.pos for point in self.col_points], dim=-1)
        blueprint_dists = torch.cat([bp.state.pos - agent.state.pos for bp in self.blueprints], dim=-1)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                col_point_dists,
                blueprint_dists
                # TODO: Add more..
                #  1. Material being carried: ID and Partner ID
            ],
            dim=-1,
        )

    def reward(self, agent: DOTSAgent) -> AGENT_REWARD_TYPE:
        # QUESTION: We just calculate a reward signal for agent 0??
        is_first = agent == self.world.agents[0]

        if is_first:
            # TODO: Replace with actual reward signal(s).
            self.pos_rew[:] = 0
            self.ground_rew[:] = 0

            # TODO: Compute the reward(s)
        # TODO: return some function of rewards (eg. rewA + rewB)
        return self.pos_rew + self.ground_rew

    def done(self):
        # TODO: This is placeholder.. Return a task completion signal.
        return torch.full(
            (self.world.batch_dim,), False, device=self.world.device
        )

    def info(self, agent: DOTSAgent) -> dict:
        # TODO: Return a dictionary of reward signals to provide debugging/logging info.
        return {"pos_rew": self.pos_rew, "ground_rew": self.ground_rew}


# TODO: Look at football example for complex agent policy implementation
class AgentPolicy:
    def __init__(self):
        # TODO: Add agent properties specific to this task
        self.initialised = False
        self.carrying = False
        self.matched = False

    def init(self, world):
        self.initialised = True
        self.world = world

        # Question: Are we building an action list for all agents, per agent..?
        self.actions = {
            agent: {
                "carrying": torch.zeros(
                    self.world.batch_dim, device=world.device
                ).bool(),  # Is the agent currently carrying a part
                "matched": torch.zeros(
                    self.world.batch_dim, device=world.device
                ).bool()  # Is the agent matched with the corresponding agent & part.
            }
            for agent in self.world.agents
        }


if __name__ == "__main__":
    render_interactively(
        __file__,
        n_agents=4,
        n_collection_points=2,
        n_blueprints=2
    )
