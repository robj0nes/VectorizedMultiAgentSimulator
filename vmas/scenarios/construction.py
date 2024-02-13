import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Line, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y, AGENT_REWARD_TYPE, AGENT_OBS_TYPE, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = kwargs.get("n_agents", 4)
        self.agent_radius = 0.1
        assert self.n_agents > 1

        self.n_collection_points = kwargs.get("n_collection_points", 2)
        self.col_point_size = 0.3
        assert self.n_collection_points > 1

        self.arena_size = 5

        # Make a default world
        world = World(batch_dim, device, collision_force=400, substeps=5)

        # Instantiate agents
        for i in range(self.n_agents):
            agent = Agent(
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
                color=Color.BLUE
            )
            self.col_points.append(col_point)
            world.add_landmark(col_point)

        self.spawn_map(world)

        # TODO: Replace with actual reward signals.
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()

        return world

    def spawn_map(self, world):
        # Instantiate floor
        self.floor = Landmark(
            name="floor",
            collide=False,
            shape=Box(length=self.arena_size, width=self.arena_size),
            color=Color.WHITE,
        )
        world.add_landmark(self.floor)

        # Instantiate Walls
        self.walls = []
        for i in range(4):
            wall = Landmark(
                name=f"wall_{i}",
                collide=True,
                shape=Box(length=self.arena_size + 0.1, width=0.1),
                color=Color.BLACK
            )
            self.walls.append(wall)
            world.add_landmark(wall)

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
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 2] if i < 2 else 0,
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        self.floor.set_pos(
            torch.tensor(
                [0, 0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

    def reset_agents(self, env_index):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.col_points,
            self.world,
            env_index,
            min_dist_between_entities=0.1,
            x_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2)),
            y_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2))
        )


    def reset_world_at(self, env_index: int = None):
        # TODO: Implement resetting the world as tensors.
        #   For each agent instantiate at a random position.
        #   Instantiate n collection points in random locations
        #   Distribute construction pieces between collection points.
        self.reset_map(env_index)
        self.reset_agents(env_index)


    def observation(self, agent: Agent) -> AGENT_OBS_TYPE:
        # TODO: Return all observables within agents FOV
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                # TODO: Add more..
            ],
            dim=-1,
        )

    def reward(self, agent: Agent) -> AGENT_REWARD_TYPE:
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
        # TODO: Return a task completion signal.
        return torch.full(
                (self.world.batch_dim,), False, device=self.world.device
            )

    def info(self, agent: Agent) -> dict:
        # TODO: Return a dictionary of reward signals to provide debugging/logging info.
        return {"pos_rew": self.pos_rew, "ground_rew": self.ground_rew}




if __name__ == "__main__":
    render_interactively(
        __file__,
        n_agents=4
    )