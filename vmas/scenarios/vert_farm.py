import math
import time
from enum import Enum
from typing import Optional, List

import numpy as np
import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator import rendering
from vmas.simulator.core import Agent, World, Landmark, Line, Sphere, Box
from vmas.simulator.dynamics.grid import Grid
from vmas.simulator.rendering import Geom
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_REWARD_TYPE, AGENT_OBS_TYPE, Color


class GridStatus(Enum):
    EMPTY = 0
    BENCH_PASSABLE = 1
    BENCH_OBSTACLE = 2
    PLANT = 3
    FLOWER = 4
    POLLINATED = 5


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.viewer_zoom = 1.5

        self.n_agents = 2
        self.agent_radius = 0.05
        self.reward_per_flower = 10
        self.shared_reward = True

        self.obs_noise = kwargs.pop("obs_noise", 0.0)
        self.grid_spacing = 0.5
        self.num_grids = 2
        self.cell_size = self.agent_radius * 4
        self.grid_size = (9, 9)
        self.grid_width = self.grid_size[0] * self.cell_size
        self.num_gridlines = self.grid_size[0]
        self.show_grid_lines = True

        self.num_benches = 2
        self.bench_coverage = 0.3
        self.bench_coords, self.bench_width, self.bench_height = self.compute_bench_positions(self.num_benches,
                                                                                              self.bench_coverage)

        self.num_plants = 20
        self.flower_prop = 0.3
        self.num_flowers = int(self.num_plants * self.flower_prop)

        # Cells are tensor: [Batch_dim, X, Y, Z, 3] ([cell center_x, cell center_y, GRIDSTATUS])
        self.cells = torch.zeros(
            (batch_dim, self.grid_size[0], self.grid_size[1], self.num_grids, 3),
            device=device
        )

        # Create basic World class according to the size and number of grids (num floors)
        world = World(
            batch_dim,
            device,
            x_semidim=self.grid_width / 2,
            y_semidim=0.5 * (
                    self.grid_width * self.num_grids +
                    self.grid_spacing * (self.num_grids - 1)),

        )

        # Add agents and append new state property: cell position. [x, y, z]
        for i in range(self.n_agents):
            agent = Agent(
                name=f'agent_{i}',
                shape=Sphere(self.agent_radius),
                obs_noise=self.obs_noise,
                render_action=True,
                action_size=4,
                dynamics=Grid(action_size=4, step=self.cell_size),
            )
            agent.state.cell_pos = torch.zeros(batch_dim, 3).to(torch.int).to(device)
            world.add_agent(agent)

        # Instantiate benches, pillars (impassable bench columns) and plants.
        #   TODO: Probably a better way to handle the pillars.
        self.benches = []
        for i in range(self.num_benches):
            for j in range(self.num_grids):
                bench = Landmark(
                    name=f'bench{i}_floor{j}',
                    collide=False,
                    shape=Box(length=(self.bench_width + 1) * self.cell_size,
                              width=(self.bench_height + 1) * self.cell_size),
                    color=(0.25, 0.25, 0.25, 0.15)
                )
                self.benches.append(bench)
                world.add_landmark(bench)

        self.pillars = []
        for i in range(self.num_benches * self.num_grids):
            bench = []
            for j in range(4):
                piller = Landmark(
                    name=f'pillar_{i}_{j}',
                    collide=True,
                    shape=Box(length=self.cell_size, width=self.cell_size),
                    color=(0.75, 0.75, 0.75, 1)
                )
                bench.append(piller)
                world.add_landmark(piller)
            self.pillars.append(bench)

        self.plants = []
        for i in range(self.num_plants):
            if i < self.num_flowers:
                flower = Landmark(
                    name=f'flower{i}',
                    shape=Box(length=self.cell_size, width=self.cell_size),
                    collide=False,
                    color=(0, 0, 0, 0)
                )
                # We want to define the flower color as a per-batch to support visualising.
                col = torch.tensor(Color.RED.value, device=device)
                flower.state.color = col.repeat(batch_dim, 1)
                flower.state.cell_pos = torch.zeros(batch_dim, 3).to(torch.int).to(device)
                self.plants.append(flower)
                world.add_landmark(flower)
            else:
                non_flower = Landmark(
                    name=f'non_flower{i - self.num_flowers}',
                    shape=Box(length=self.cell_size, width=self.cell_size),
                    collide=False,
                    color=(0, 0, 0, 0)
                )
                col = torch.tensor(Color.BLUE.value, device=device)
                non_flower.state.color = col.repeat(batch_dim, 1)
                non_flower.state.cell_pos = torch.zeros(batch_dim, 3).to(torch.int).to(device)
                self.plants.append(non_flower)
                world.add_landmark(non_flower)

        # Set global rewards
        self.rewards = torch.zeros(batch_dim, device=device)

        # Create the elements which will form the grid visualisation.
        self.grids = {
            "walls": [],
            "lines": [],
        }
        for i in range(self.num_grids):
            walls = []
            for j in range(4):
                wall = Landmark(
                    name=f"grid {i} wall {j}",
                    collide=True,
                    shape=Line(length=self.grid_width),
                    color=Color.BLACK,
                )
                world.add_landmark(wall)
                walls.append(wall)
            self.grids['walls'].append(walls)

            lines = []
            for j in range(self.num_gridlines * 2):
                line = Landmark(
                    name=f'grid {i} line {j}',
                    collide=False,
                    shape=Line(length=self.grid_width),
                    color=(0.25, 0.25, 0.25, 0.25),
                )
                world.add_landmark(line)
                lines.append(line)
            self.grids['lines'].append(lines)

        return world

    def reset_world_at(self, env_index: Optional[int] = None):
        self.reset_cells(env_index)
        # Build the world
        self.spawn_grids(env_index)
        self.spawn_benches(env_index)
        self.spawn_flowers(env_index)
        self.spawn_agents(env_index)

    def done(self) -> Tensor:
        pollinated = torch.Tensor(self.cells[:, :, :, :, -1] == GridStatus.POLLINATED.value)
        test = pollinated.sum(dim=tuple(range(1, pollinated.ndim))) == self.num_flowers
        return pollinated.sum(dim=tuple(range(1, pollinated.ndim))) == self.num_flowers

    def observation(self, agent: Agent) -> AGENT_OBS_TYPE:
        """
        Returns agent cell position, plant cell positions and plant colours (ie. states)
        """
        return torch.cat(
            [agent.state.cell_pos] +
            [f.state.cell_pos for f in self.plants] +
            [f.state.color for f in self.plants]
            , dim=-1
        )

    # Per-step reward function. Currently only shared rewards implemented.
    def reward(self, agent: Agent) -> AGENT_REWARD_TYPE:
        if self.shared_reward:
            """
            Shared reward: 
                Reward returned for every currently pollinated flower in the world. 
            """
            is_first = agent == self.world.agents[0]
            if is_first:
                self.rew = torch.zeros(
                    self.world.batch_dim,
                    device=self.world.device,
                    dtype=torch.float32,
                )
                pollinated = torch.Tensor(self.cells[:, :, :, :, -1] == GridStatus.POLLINATED.value)
                self.rew += pollinated.sum(dim=tuple(range(1, pollinated.ndim))) * self.reward_per_flower

        else:
            # TODO: Handle independent agent reward function.
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
            # dist_to_goal = torch.linalg.vector_norm(
            #     agent.state.pos - agent.goal.state.pos, dim=1
            # )
            # agent_shaping = dist_to_goal * self.shaping_factor
            # self.rew += agent.global_shaping - agent_shaping
            # agent.global_shaping = agent_shaping

        return self.rew

    def check_vertical_movement(self, curr_cell: torch.Tensor, next_cell: torch.Tensor,
                                vert_move: torch.Tensor) -> torch.tensor:
        result_mask = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)

        cstate = self.cells[vert_move, curr_cell[vert_move, 0], curr_cell[vert_move, 1], curr_cell[vert_move, 2], -1]
        nstate = self.cells[vert_move, next_cell[vert_move, 0], next_cell[vert_move, 1], next_cell[vert_move, 2], -1]
        cmask = torch.Tensor((cstate == GridStatus.BENCH_PASSABLE.value) |
                             (cstate == GridStatus.PLANT.value) |
                             (cstate == GridStatus.FLOWER.value))
        nmask = torch.Tensor((nstate == GridStatus.BENCH_PASSABLE.value) |
                             (nstate == GridStatus.PLANT.value) |
                             (nstate == GridStatus.FLOWER.value))
        active_mask = torch.logical_and(cmask, nmask)
        result_mask[vert_move] = active_mask
        return result_mask

    def handle_movement(self, agent, action):
        movement = action[:, :3].to(torch.int)
        next_cell = (agent.state.cell_pos + movement).to(torch.int)

        # Clamp next movement to within bounds of grid.
        ncx = torch.clamp(next_cell[:, 0].to(torch.int), min=0, max=self.grid_size[0] - 1)
        ncy = torch.clamp(next_cell[:, 1].to(torch.int), min=0, max=self.grid_size[1] - 1)
        ncz = torch.clamp(next_cell[:, 2].to(torch.int), min=0, max=self.num_grids - 1)
        next_cell = torch.stack([ncx, ncy, ncz], -1)

        vertical_moves = torch.not_equal(agent.state.cell_pos[:, -1], next_cell[:, -1])
        horizontal_moves = ~vertical_moves
        illegal_moves = torch.zeros(next_cell.shape[0], dtype=torch.bool, device=self.world.device)

        # Note: Check this with more than 1 env.
        if torch.any(vertical_moves):
            illegal_moves = illegal_moves | self.check_vertical_movement(agent.state.cell_pos,
                                                                         next_cell, vertical_moves)

        if torch.any(horizontal_moves):
            result_mask = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
            next_state = self.cells[horizontal_moves, next_cell[horizontal_moves, 0],
                                    next_cell[horizontal_moves, 1], next_cell[horizontal_moves, 2], -1]
            next_mask = next_state == GridStatus.BENCH_OBSTACLE.value
            result_mask[horizontal_moves] = next_mask
            illegal_moves = illegal_moves | result_mask

        if torch.any(illegal_moves):
            # TODO: Apply (per-agent) negative reward to batches with illegal moves.
            legal_moves = ~illegal_moves
            agent.state.pos[legal_moves] = self.cells[legal_moves, ncx[legal_moves], ncy[legal_moves], ncz[legal_moves], :2].squeeze(1)
            agent.state.cell_pos[legal_moves] = next_cell[legal_moves]
        else:
            batch_indices = torch.arange(self.world.batch_dim)
            agent.state.pos = self.cells[batch_indices, ncx, ncy, ncz, :2]
            agent.state.cell_pos = next_cell

    def handle_pollination(self, agent, action):
        # TODO: We probably want to apply negative reward to agents trying to pollinate
        #  non-flowering plants and cells which are not plants.

        pollinate = action[:, 3].to(torch.int)  # Get batches where agent action is to 'pollinate'
        c_pos = agent.state.cell_pos
        batch_indices = torch.arange(self.world.batch_dim, device=self.world.device)

        on_flower = (self.cells[batch_indices, c_pos[:, 0], c_pos[:, 1], c_pos[:, 2], -1] == GridStatus.FLOWER.value)  # Get batches where agent pos == flower pos
        pollinated = torch.logical_and(pollinate, on_flower)
        # pollinated = (pollinate & on_flower).squeeze(-1).to(torch.bool)  # Mask for <on_flower> && <pollinate action>

        if pollinated.any():
            # Update cell status and flower colour
            self.cells[pollinated, c_pos[pollinated, 0], c_pos[pollinated, 1], c_pos[
                pollinated, 2], -1] = GridStatus.POLLINATED.value
            for flower in self.plants:
                if torch.equal(flower.state.pos[pollinated],
                               self.cells[pollinated, c_pos[pollinated, 0], c_pos[pollinated, 1],
                               c_pos[pollinated, 2], :2]):
                    flower.state.color[pollinated] = torch.tensor(Color.GREEN.value)

    def process_action(self, agent: Agent):
        action = agent.action.u
        if not action.abs().sum().item() == 0:
            self.handle_movement(agent, action)
            self.handle_pollination(agent, action)

    def reset_cells(self, env_index):
        if env_index is not None:
            self.cells[:] = 0
        else:
            self.cells[:] = 0

    def spawn_agents(self, env_index):
        for agent in self.world.agents:
            if env_index is not None:
                # Find a random free cell and update agent position.
                free_cells = torch.nonzero(self.cells[env_index, :, :, :, -1] == 0, as_tuple=False)
                choice = free_cells[torch.randint(0, free_cells.size(0), (1,)).item()]
                agent.state.pos[env_index] = self.cells[env_index, choice[0], choice[1], choice[2], :2]
                agent.state.cell_pos[env_index] = choice.to(torch.int)
            else:
                for b in range(self.world.batch_dim):
                    # Find a random free cell and update agent position.
                    free_cells = torch.nonzero(self.cells[b, :, :, :, -1] == 0, as_tuple=False)
                    choice = free_cells[torch.randint(0, free_cells.size(0), (1,)).item()]
                    agent.state.pos[b] = self.cells[b, choice[0], choice[1], choice[2], :2]
                    agent.state.cell_pos[b] = choice

    def spawn_grids(self, env_index):
        """
        Handles the placing of non-interactive visual elements: walls, gridlines
        """
        for j, grid in enumerate(self.grids):
            self.spawn_walls(env_index, j)
            self.locate_cells(env_index, j)
            if self.show_grid_lines:
                self.spawn_grid_lines(env_index, j)

    def spawn_walls(self, env_index, grid_index):
        y_offset = (self.grid_width + self.grid_spacing) * grid_index

        for i, wall in enumerate(self.grids['walls'][grid_index]):
            wall.set_pos(
                torch.tensor(
                    [
                        (
                            0.0
                            if i % 2
                            else (
                                self.world.x_semidim
                                if i == 0
                                else -self.world.x_semidim
                            )
                        ),
                        (
                            -self.world.y_semidim + self.grid_width / 2 + y_offset
                            if not i % 2
                            else (
                                -self.world.y_semidim + self.grid_width + y_offset
                                if i == 1
                                else -self.world.y_semidim + y_offset
                            )
                        ),
                    ],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            wall.set_rot(
                torch.tensor(
                    [torch.pi / 2 if not i % 2 else 0.0],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def compute_bench_positions(self, num_benches, bench_coverage):
        """
        returns [x_min, x_max, y_min, y_max] for each bench in one layer assuming they are the same in all layers.
        """
        area_per_bench = (bench_coverage * np.prod(self.grid_size)) / num_benches

        bench_width = int(np.sqrt(area_per_bench) / 1.5)
        bench_height = int(area_per_bench / bench_width)

        # Layout benches in rows and cols
        rows = int(math.floor(math.sqrt(num_benches)))
        cols = int(math.ceil(num_benches / rows))

        # Compute horizontal and vertical spacing
        x_spacing = int((self.grid_size[0] - cols * bench_width) / (cols + 1))
        y_spacing = int((self.grid_size[1] - rows * bench_height) / (rows + 1))

        positions = []
        # for _ in range(num_benches):num_benches
        for i in range(rows):
            for j in range(cols):
                x = x_spacing + j * (bench_width + x_spacing + 1 if x_spacing == 1 else 0)
                y = y_spacing + i * (bench_height + y_spacing)
                positions.append((x, y))
        bench_coords = [
            [p[0], p[0] + bench_width, p[1], p[1] + bench_height] for p in positions
        ]

        return bench_coords, bench_width, bench_height

    def spawn_benches(self, env_index):
        for i, bench in enumerate(self.benches):
            coords = self.bench_coords[i % self.num_benches]
            center = int((coords[1] + coords[0]) / 2), int((coords[3] + coords[2]) / 2)
            if env_index is not None:
                bench.state.pos[env_index] = self.cells[env_index, center[0], center[1], i // self.num_benches, :2]
            else:
                bench.state.pos = self.cells[:, center[0], center[1], i // self.num_benches, :2]
            for x in range(coords[0], coords[1] + 1):
                for y in range(coords[2], coords[3] + 1):
                    if env_index is not None:
                        self.cells[env_index, x, y, i // self.num_benches, -1] = GridStatus.BENCH_PASSABLE.value
                    else:
                        self.cells[:, x, y, i // self.num_benches, -1] = GridStatus.BENCH_PASSABLE.value

        for i, bench in enumerate(self.pillars):
            coords = self.bench_coords[i % self.num_benches]
            for j, p in enumerate(bench):
                if j == 0:
                    x, y = coords[0], coords[2]
                elif j == 1:
                    x, y = coords[0], coords[3]
                elif j == 2:
                    x, y = coords[1], coords[2]
                else:
                    x, y = coords[1], coords[3]
                if env_index is not None:
                    p.state.pos[env_index] = self.cells[env_index, x, y, i // self.num_benches, :2]
                    self.cells[env_index, x, y, i // self.num_benches, -1] = GridStatus.BENCH_OBSTACLE.value
                else:
                    p.state.pos = self.cells[:, x, y, i // self.num_benches, :2]
                    self.cells[:, x, y, i // self.num_benches, -1] = GridStatus.BENCH_OBSTACLE.value

    def spawn_flowers(self, env_index):
        if env_index is not None:
            for i in range(self.num_plants):
                # pick a random_bench
                min_x, max_x, min_y, max_y = self.bench_coords[np.random.randint(len(self.bench_coords))]

                occupied = True
                while occupied:
                    # Random x, y within the bounds
                    x = np.random.randint(min_x, max_x + 1)
                    y = np.random.randint(min_y, max_y + 1)
                    z = np.random.randint(0, self.num_grids)
                    occupied = self.check_empty_cell(env_index, x, y, z)

                self.plants[i].state.pos[env_index] = self.cells[env_index, x, y, z, :2]
                self.plants[i].state.cell_pos[env_index] = torch.tensor([x, y, z])
                self.plants[i].state.color[env_index] = torch.tensor(Color.RED.value) \
                    if i < self.num_flowers else torch.tensor(Color.BLUE.value)
                self.cells[env_index, x, y, z, -1] = GridStatus.FLOWER.value \
                    if i < self.num_flowers else GridStatus.PLANT.value

        else:
            for b in range(self.world.batch_dim):
                for i in range(self.num_plants):
                    # pick a random_bench
                    min_x, max_x, min_y, max_y = self.bench_coords[np.random.randint(len(self.bench_coords))]

                    occupied = True
                    while occupied:
                        # Random x, y within the bounds
                        x = np.random.randint(min_x, max_x + 1)
                        y = np.random.randint(min_y, max_y + 1)
                        z = np.random.randint(0, self.num_grids)
                        occupied = self.check_empty_cell(b, x, y, z)

                    self.plants[i].state.pos[b] = self.cells[b, x, y, z, :2]
                    self.plants[i].state.cell_pos[b] = torch.tensor([x, y, z])
                    self.plants[i].state.color[b] = torch.tensor(Color.RED.value) \
                        if i < self.num_flowers else torch.tensor(Color.BLUE.value)
                    self.cells[b, x, y, z, -1] = GridStatus.FLOWER.value \
                        if i < self.num_flowers else GridStatus.PLANT.value

    def locate_cells(self, env_index, grid_index):
        y_offset = (self.grid_width + self.grid_spacing) * grid_index

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                x_cent = -self.world.x_semidim + (self.cell_size * x) + self.cell_size / 2
                y_cent = -self.world.y_semidim + y_offset + (self.cell_size * y) + self.cell_size / 2

                if env_index is not None:
                    self.cells[env_index, x, y, grid_index] = torch.tensor([
                        x_cent, y_cent,
                        0,
                    ])
                else:
                    self.cells[:, x, y, grid_index] = torch.tensor([
                        x_cent, y_cent,
                        0,
                    ])

    def check_empty_cell(self, env_index, x, y, z):
        if env_index is not None:
            return not (self.cells[env_index, x, y, z, -1] == 0 or self.cells[
                env_index, x, y, z, -1] == GridStatus.BENCH_PASSABLE.value)
        else:
            return not (self.cells[:, x, y, z, -1] == 0 or self.cells[
                env_index, x, y, z, -1] == GridStatus.BENCH_PASSABLE.value)

    def spawn_grid_lines(self, env_index, grid_index):
        y_offset = (self.grid_width + self.grid_spacing) * grid_index
        for i, line in enumerate(self.grids['lines'][grid_index]):
            line.set_pos(
                torch.tensor(
                    [
                        (
                            0.0
                            if i > self.num_gridlines - 1
                            else (
                                    -self.world.x_semidim + (self.cell_size * i)
                            )
                        ),
                        (
                            -self.world.y_semidim + self.grid_width / 2 + y_offset
                            if not i >= self.num_gridlines
                            else (
                                    -self.world.y_semidim + y_offset +
                                    (self.cell_size * (i - self.num_gridlines))
                            )
                        ),
                    ],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            line.set_rot(
                torch.tensor(
                    [torch.pi / 2 if i < self.num_gridlines else 0.0],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        geoms = []
        for plant in self.plants:
            c_pos = plant.state.cell_pos[env_index]
            x_cent = self.cells[env_index, c_pos[0], c_pos[1], c_pos[2], 0]
            y_cent = self.cells[env_index, c_pos[0], c_pos[1], c_pos[2], 1]

            bench_cell = rendering.make_polygon(v=[
                [x_cent - self.cell_size / 2, y_cent - self.cell_size / 2],
                [x_cent - self.cell_size / 2, y_cent + self.cell_size / 2],
                [x_cent + self.cell_size / 2, y_cent + self.cell_size / 2],
                [x_cent + self.cell_size / 2, y_cent - self.cell_size / 2],
            ])
            bench_cell.set_color(*tuple(plant.state.color[env_index].tolist()))
            geoms.append(bench_cell)
        return geoms


if __name__ == "__main__":
    rnd_seed = int(time.time())
    render_interactively(__file__, control_two_agents=True, discrete_control=True, seed=rnd_seed, action_size=4)
