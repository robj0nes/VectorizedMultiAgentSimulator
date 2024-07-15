import math
import random
import numpy as np
import torch
import seaborn as sns

from vmas import render_interactively
from vmas.simulator import rendering
from vmas.simulator.core import Sphere, World, Box
from vmas.simulator.dots_core import DOTSWorld, DOTSAgent, DOTSComsNetwork, DOTSPayloadDest
from vmas.simulator.rendering import TextLine
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_REWARD_TYPE, AGENT_OBS_TYPE, ScenarioUtils, Y, X, Color

# TODO:
#  1. Mixing: Dont freeze mixing when goal is reached....
#  2. Multi-goal setting:
#       - Re-implement goal selection mechanisms and test 3:3 scenario performs the same.
#       - Need a computationally efficient way of identifying dynamic target-goal index in batch dimensions.

EPSILON = 1e-6


def get_distinct_color_list(n_cols, device: torch.device, exclusions=None):
    opts = sns.color_palette(palette="Set2")
    return torch.stack([torch.tensor(opts[x], device=device, dtype=torch.float32) for x in
                        random.sample(range(0, len(opts)), n_cols)])


# TODO (non urgent): Flip tensors so that the batch dim is always first for consistency with VMAS implementation.

class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        # World properties
        self.n_agents = None
        self.n_goals = None
        self.arena_size = None
        self.task_type = None
        self.goals = None
        self.selected_goals = None
        self.completed_goals = None
        self.agent_list = None
        self.all_agents = None
        self.coms_network = None

        # Agent properties
        self.knowledge_shape = None
        self.random_knowledge = None
        self.multi_head = None
        self.agent_radius = None
        self.dim_c = None
        self.agent_action_size = None

        # Observation and reward properties
        self.observe_other_agents = None
        self.observe_all_goals = None
        self.isolated_coms = None
        self.coms_proximity = None
        self.observation_proximity = None
        self.mixing_thresh = 0.01
        self.learn_mix = None
        self.learn_coms = None
        self.per_goal_reward = None
        self.reward_multiplier = 1

    # ----------- World Creation ----------- #

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.initialise_task_properties(kwargs)
        self.initialise_reward_properties(batch_dim, device, kwargs)

        world = DOTSWorld(batch_dim, device, collision_force=100, dim_c=self.dim_c)

        self.instantiate_goals(batch_dim, device, world)
        self.instantiate_agents(batch_dim, device, world)

        world.spawn_map()

        return world

    def initialise_task_properties(self, kwargs):
        self.task_type = kwargs.get("task_type", "nav")
        self.debug_goals = kwargs.get("debug_goals", False)
        self.n_agents = kwargs.get("n_agents", 4)
        self.n_goals = kwargs.get("n_goals", 4)
        self.agent_radius = 0.2
        self.arena_size = 5
        self.viewer_zoom = 1.7
        # Knowledge is of shape: [2 (source, learnt), knowledge dim (eg. 3-RGB)]
        self.knowledge_shape = kwargs.get("knowledge_shape", (2, 3))
        self.random_knowledge = kwargs.get("random_knowledge", False)
        self.multi_head = kwargs.get("multi_head", False)
        self.observation_proximity = kwargs.get("observation_proximity", self.arena_size)
        self.observe_all_goals = kwargs.get("observe_all_goals", False)
        self.observe_other_agents = kwargs.get("observe_other_agents", True)
        self.isolated_coms = kwargs.get("isolated_coms", False)
        self.coms_proximity = kwargs.get("coms_proximity", self.arena_size)
        self.learn_coms = kwargs.get("learn_coms", True)
        self.mixing_thresh = kwargs.get("mixing_thresh", 0.01)
        self.learn_mix = kwargs.get("learn_mix", True)
        # Size of coms = No coms if only testing navigation,
        #   else: 1-D mixing intent + N-D knowledge.
        self.dim_c = kwargs.get("dim_c", 1 + self.knowledge_shape[1]) \
            if self.task_type != 'nav' else 0
        # Size of action = 2-D force + N-D knowledge co-efficients.
        self.agent_action_size = kwargs.get(
            "action_size",
            2 + self.knowledge_shape[1] if self.task_type != 'nav' else 2)

    def initialise_reward_properties(self, batch_dim, device, kwargs):
        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -0.2)
        self.env_collision_penalty = kwargs.get("env_collision_penalty", -0.2)
        self.min_collision_distance = kwargs.get("collision_dist", 0.005)
        self.pos_shaping = kwargs.get("pos_shaping", False)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1.0)
        self.mix_shaping = kwargs.get("mix_shaping", False)
        self.mix_shaping_factor = kwargs.get("mix_shaping_factor", 1.0)
        self.default_task_rew = kwargs.get("default_task_rew", 0.02)
        self.final_pos_reward = kwargs.get("final_pos_reward", 0.05)
        self.final_mix_reward = kwargs.get("final_mix_reward", 0.05)
        self.per_agent_reward = kwargs.get("per_agent_reward", False)
        self.per_goal_reward = kwargs.get("per_goal_reward", 0.1)
        self.final_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.final_pos_rew = self.final_rew.clone()
        self.final_mix_rew = self.final_rew.clone()

    def instantiate_agents(self, batch_dim, device, world):
        self.all_agents = []
        self.agent_list = {"nav": [], "mix": []}
        # TODO: If we take this approach we need to consider how to implement the two agents as one entity.
        # Question: Can we dynamically create agent groups?
        # Handle multi-head agent naming conventions.
        if self.multi_head:
            name_ext = ["nav-", "mix-"]
            for ext in name_ext:
                for i in range(self.n_agents):
                    agent = DOTSAgent(
                        name=f"{ext}agent_{i}",
                        shape=Sphere(self.agent_radius) if "nav" in ext else None,
                        collide=True if "nav" in ext else False,
                        color=Color.GREEN,
                        agent_index=i,
                        task=ext.strip('-'),
                        render=True if "nav" in ext else False,
                        knowledge_shape=None if "nav" in ext else self.knowledge_shape,
                        silent=True if self.dim_c == 0 else False,
                        # silent=True if "nav" in ext or self.dim_c == 0 else False,
                        action_size=self.agent_action_size
                    )
                    if "nav" in ext:
                        self.selected_goals[i, :] = True
                        agent.rewards = {
                            "agent_collision": torch.zeros(batch_dim, device=device),
                            "obstacle_collision": torch.zeros(batch_dim, device=device),
                            "position": torch.zeros(batch_dim, device=device),
                            "final": torch.zeros(batch_dim, device=device)
                        }
                    if "mix" in ext:
                        agent.rewards = {
                            "mixing": torch.zeros(batch_dim, device=device),
                            "final": torch.zeros(batch_dim, device=device)
                        }
                    self.agent_list[ext.strip("-")].append(agent)
                    self.all_agents.append(agent)
                    world.add_agent(agent)
        else:
            for i in range(self.n_agents):
                agent = DOTSAgent(
                    name=f"agent_{i}",
                    shape=Sphere(self.agent_radius),
                    color=Color.GREEN,
                    task=self.task_type,
                    agent_index=i,
                    knowledge_shape=self.knowledge_shape,
                    silent=True if self.dim_c == 0 else False,
                    action_size=self.agent_action_size
                )
                # TODO: Refactor to nested dictionary/object? Or make part of agent state?
                agent.rewards = {
                    "agent_collision": torch.zeros(batch_dim, device=device),
                    "obstacle_collision": torch.zeros(batch_dim, device=device),
                    "position": torch.zeros(batch_dim, device=device),
                    "mixing": torch.zeros(batch_dim, device=device),
                    "final": torch.zeros(batch_dim, device=device)
                }

                self.selected_goals[i, :] = True
                self.agent_list["nav"].append(agent)
                self.agent_list["mix"].append(agent)
                self.all_agents.append(agent)
                world.add_agent(agent)
        if self.isolated_coms:
            # Learn a separate coms network which observes all agent coms signals
            #  and outputs coms of size [n_agents * dim_c]
            coms_network = DOTSComsNetwork(
                name="coms_network",
                action_size=self.dim_c * self.n_agents
            )
            coms_network.final_rew = torch.zeros(batch_dim, device=device)
            world.add_agent(coms_network)
            self.coms_network = coms_network

    def instantiate_goals(self, batch_dim, device, world):
        self.goals = []
        self.selected_goals = torch.stack(
            [torch.zeros(batch_dim, device=device, dtype=torch.bool) for _ in range(self.n_goals)]
        )
        self.completed_goals = self.selected_goals.clone()
        for i in range(self.n_goals):
            goal = DOTSPayloadDest(
                name=f"goal_{i}",
                collide=False,
                shape=Box(length=self.agent_radius * 4, width=self.agent_radius * 4),
                color=Color.BLUE,
                expected_knowledge_shape=3
            )
            self.goals.append(goal)
            world.add_landmark(goal)

    # ----------- Knowledge Creation ----------- #

    def get_preset_knowledge(self, small, large, device):
        large_knowledge = torch.stack(
            [
                get_distinct_color_list(large, device) for _ in range(self.world.batch_dim)
            ]
        )
        # NOTE: Used to debug goal switching behaviours
        # large_knowledge[:, self.n_agents:self.n_agents * 2, :] = large_knowledge[:, :self.n_agents, :]
        # large_knowledge[:, self.n_agents * 2:] = large_knowledge[:, :2, :]

        # ran_ints = [
        #     random.sample(range(large_knowledges.shape[1]), small) for _ in range(self.world.batch_dim)
        # ]

        # TODO: Test nav with random agent-goal assignments. Make sure that observations are
        #  based on colour match, not agent index.
        small_knowledge = torch.stack(
            [
                torch.stack(
                    [
                        # Random goal assignment
                        # large_knowledges[j][i] for i in ran_ints[j]

                        # Indexed goal assignment
                        large_knowledge[j][i] for i in range(large_knowledge.shape[1])
                    ]
                )
                for j in range(self.world.batch_dim)
            ]
        )

        for i in range(self.world.batch_dim):
            for sp in small_knowledge[i]:
                assert (
                        sp in large_knowledge[i]
                ), "Trying to assign matching knowledges to goals, but found a mismatch"

        return small_knowledge, large_knowledge

    # TODO: Have some check to ensure all goal mixes are unique.
    def generate_knowledge(self, device):
        if self.random_knowledge:
            # Generate a set of random knowledge vectors K (one per agent) such that sum over all knowledge K_i > 1.
            agent_knowledge, solvable = None, False
            while not solvable:
                agent_knowledge = torch.stack(
                    [
                        torch.stack([
                            torch.tensor(np.random.uniform(0.01, 1.0, self.knowledge_shape[1]), device=device,
                                         dtype=torch.float32)
                            for _ in range(self.n_agents)
                        ])
                        for _ in range(self.world.batch_dim)
                    ]
                )
                solvable = True
                for dim in range(self.knowledge_shape[1]):
                    if torch.sum(agent_knowledge[:, dim]) < 1:
                        solvable = False
                        print("Regenerating agent knowledge to avoid unsolvable problem")

        else:
            single_batch = torch.stack(
                [torch.zeros(self.n_agents, device=device, dtype=torch.float32) for _ in range(self.n_agents)])
            for i in range(self.n_agents):
                single_batch[i][i] += 1
            agent_knowledge = torch.stack(
                [
                    single_batch.clone()
                    for _ in range(self.world.batch_dim)
                ]
            )

        # Generate a random solution of knowledge size for each goal.
        goal_knowledge = torch.stack(
            [
                torch.stack([
                    torch.tensor(np.random.uniform(0.01, 1.0, self.knowledge_shape[1]), device=device,
                                 dtype=torch.float32)
                    for _ in range(self.n_goals)
                ])
                for _ in range(self.world.batch_dim)
            ]
        )

        if agent_knowledge is None:
            raise ValueError("Agent knowledge is None")

        return agent_knowledge, goal_knowledge

    def random_paint_generator(self, device: torch.device):
        if self.task_type == 'nav' or self.debug_goals:
            if self.n_goals <= self.n_agents:
                goal_knowledge, agent_knowledge = self.get_preset_knowledge(self.n_goals, self.n_agents, device)
            else:
                agent_knowledge, goal_knowledge = self.get_preset_knowledge(self.n_agents, self.n_goals, device)
        else:
            agent_knowledge, goal_knowledge = self.generate_knowledge(device)
        return agent_knowledge, goal_knowledge

    # -------- Env Resets -------------#

    def reset_agents_and_goals(self, env_index):
        ScenarioUtils.spawn_entities_randomly(
            self.agent_list['nav'] + self.goals,
            self.world,
            env_index,
            min_dist_between_entities=1,
            x_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2)),
            y_bounds=(int(-self.arena_size / 2), int(self.arena_size / 2))
        )

        agent_knowledge, goal_knowledge = self.random_paint_generator(self.world.device)

        for i, agent in enumerate(self.agent_list["mix"]):
            # Form a knowledge of shape [Batch_dim, RGB source, RGB mixed] default mixed to source value.
            knowledge = agent_knowledge[:, i, :].unsqueeze(1).repeat(1, 2, 1)
            if env_index is None:
                agent.set_knowledge(knowledge, batch_index=env_index)
            else:
                agent.set_knowledge(knowledge[env_index], batch_index=env_index)

        for i, goal in enumerate(self.goals):
            # Goal should be of shape: [Batch dim, RGB]
            if env_index is None:
                goal.set_expected_knowledge(goal_knowledge[:, i, :], batch_index=env_index)
            else:
                goal.set_expected_knowledge(goal_knowledge[env_index, i, :], batch_index=env_index)

        self.initialise_reward_shaping(env_index)
        if env_index is None:
            self.selected_goals[: self.n_agents] = True
            self.selected_goals[self.n_agents:] = False
            self.completed_goals[:] = False
        else:
            self.selected_goals[:self.n_agents, env_index] = True
            self.selected_goals[self.n_agents:, env_index] = False
            self.completed_goals[:, env_index] = False

    def initialise_reward_shaping(self, env_index):
        # Preliminary position and mix shaping to all goals.
        # Clone distances to use as normalisation during reward computation.
        for agent in self.agent_list["nav"]:
            if env_index is None:
                agent.pos_shaping = (
                        torch.stack(
                            [torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=-1) for goal in
                             self.goals])
                        * self.pos_shaping_factor)
                agent.pos_shape_norm = agent.pos_shaping.clone() + EPSILON
            else:
                agent.pos_shaping[:, env_index] = (
                        torch.stack(
                            [torch.linalg.vector_norm((agent.state.pos[env_index] - goal.state.pos[env_index]), dim=-1)
                             for goal in self.goals])
                        * self.pos_shaping_factor
                )
                agent.pos_shape_norm[:, env_index] = agent.pos_shaping[:, env_index] + EPSILON

        for agent in self.agent_list["mix"]:
            if env_index is None:
                agent.mix_shaping = (
                        torch.stack(
                            [
                                torch.linalg.vector_norm(
                                    (agent.state.knowledge[:, 1, :] - goal.state.expected_knowledge),
                                    dim=-1)
                                for goal in self.goals
                            ])
                        * self.mix_shaping_factor)
                agent.mix_shaping_norm = agent.mix_shaping.clone() + EPSILON
            else:
                agent.mix_shaping[:, env_index] = (
                        torch.stack(
                            [
                                torch.linalg.vector_norm(
                                    (agent.state.knowledge[env_index, 1, :] - goal.state.expected_knowledge[env_index]),
                                    dim=-1)
                                for goal in self.goals
                            ])
                        * self.mix_shaping_factor)
                agent.mix_shaping_norm[:, env_index] = agent.mix_shaping[:, env_index] + EPSILON

    def reset_world_at(self, env_index: int = None):
        self.reset_agents_and_goals(env_index)
        self.world.reset_map(env_index)

    # TODO(s):
    #  1. Implement a restriction on observations based on the self.observation_proximity value.
    #       - Maybe look at how lidar is implemented for hints on variable observation space..

    # -------------- Observations ------------ #

    def observation(self, agent: DOTSAgent) -> AGENT_OBS_TYPE:
        if type(agent).__name__ == "DOTSAgent":
            # TODO: Test if adding this reduces collisions/improves training.. first try suggests not.
            # Get vector norm distance to all other agents.
            if self.observe_other_agents and agent.task == "nav":
                other_agents = torch.transpose(
                    torch.stack(
                        [
                            torch.linalg.vector_norm((agent.state.pos - a.state.pos), dim=1)
                            for a in self.agent_list['nav'] if a != agent
                        ]
                    ), 0, 1
                )
            else:
                other_agents = torch.empty(0)

            task_obs = [self.compute_goal_observations(agent)]

            if agent.task != "nav":
                self.compute_coms_observations(agent, task_obs)

            if agent.task == "nav":
                self_obs = [agent.state.pos, agent.state.vel]
            elif agent.task == "mix":
                self_obs = [agent.state.knowledge[:, 0, :], agent.state.knowledge[:, 1, :]]
            else:
                self_obs = [agent.state.pos, agent.state.vel,
                            agent.state.knowledge[:, 0, :], agent.state.knowledge[:, 1, :]]

            return torch.cat(
                (self_obs
                 + [*task_obs]
                 + [other_agents]),
                dim=-1,
            )

        elif type(agent).__name__ == "DOTSComsNetwork":
            # Independent coms network observes all agent coms and learns to relay a signal to each of the agents.
            agent_coms = [a.state.c for a in self.agent_list["mix"]]
            return torch.cat(agent_coms, dim=-1)

    def compute_coms_observations(self, agent, task_obs):
        if self.isolated_coms:
            # Get the (agent index * dim_c) output from the coms GNN
            if self.coms_network.action.u is not None:
                agent_index = int(agent.name.split('_')[-1])
                com_start = self.dim_c * agent_index
                incoming_coms = self.coms_network.action.u[:, com_start: com_start + self.dim_c]
                task_obs.append(incoming_coms)
            else:
                task_obs.append(torch.zeros((self.world.batch_dim, self.dim_c), device=self.world.device))

        else:
            if self.learn_coms:
                # Collect communications from other agents.
                agent_coms = torch.stack(
                    [a.state.c for a in self.agent_list['mix'] if a != agent]
                )
            else:
                # Assume agents communicate correctly:
                #   Directly observe the source knowledge of the other agents.
                agent_coms = torch.stack(
                    [a.state.knowledge[:, 0, :] for a in self.agent_list['mix'] if a != agent]
                )

            for c in agent_coms:
                task_obs.append(c)

    # TODO: The full implementation for this will require some work as we need to define some variable
    #  observation of whether a knowledge-goal colour match has been made in a given batch dim.
    #   - Perhaps best to always observe goal, but only receive positional rewards once colour match has been made.
    #  ..
    def compute_goal_observations(self, agent):
        # Forms a list of tensors [distX, distY, R, G, B] for each goal.
        #   Shape [Batch size, n_goals * (2 + knowledge_dims)]
        if self.observe_all_goals:
            # If we observe all goals, simply compute distance to each goal.
            goals = [
                torch.cat(
                    [
                        goal.state.pos - agent.state.pos,
                        goal.state.expected_knowledge
                    ],
                    dim=-1)
                for goal in self.goals
            ]

        else:
            agent_index = int(agent.name.split('_')[-1])
            if agent.task == "mix":
                all_goal_knowledge = torch.stack([g.state.expected_knowledge for g in self.goals])

                target_goal_knowledge = all_goal_knowledge[agent_index]
                # target_goal_knowledge = torch.stack(
                #     [all_goal_knowledge[agent.state.target_goal[i], i, :] for i in range(len(agent.state.target_goal))]
                # )
                goals = torch.cat(
                    [target_goal_knowledge - agent.state.knowledge[:, 1, :]]
                )
            elif agent.task == "nav":
                # Goal obs is relative position.
                all_goal_pos = torch.stack([g.state.pos for g in self.goals])

                target_goal_pos = all_goal_pos[agent_index]
                # target_goal_pos = torch.stack(
                #     [all_goal_pos[agent.state.target_goal[i], i, :] for i in range(len(agent.state.target_goal))]
                # )
                goals = torch.cat(
                    [target_goal_pos - agent.state.pos],
                    dim=-1)
            else:
                # Goal obs is relative position and knowledge.
                goals = torch.cat(
                    [
                        self.goals[agent.state.target_goal].state.pos - agent.state.pos,
                        self.goals[agent.state.target_goal].state.expected_knowledge
                    ],
                    dim=-1)

        return goals

    # --------------- Rewards -----------------#

    def reward(self, agent: DOTSAgent) -> AGENT_REWARD_TYPE:
        if agent == self.coms_network:
            # TODO: Consider if we want to implement some sort of reward signal for the coms network..
            return self.final_rew

        else:
            self.reset_agent_rewards(agent)
            if agent == self.all_agents[0]:
                self.reset_final_rewards()

            self.compute_collision_penalties(agent)

            # TODO: Once confirmed that this implementation is working, test new target goal calcs.
            self.compute_positional_rewards(agent)
            self.compute_mixing_rewards(agent)

            if agent == self.all_agents[0]:
                self.compute_final_rewards()

            self.set_new_goal(agent)

            task_reward = self.final_pos_rew if agent.task == "nav" \
                else self.final_mix_rew if agent.task == "mix" \
                else torch.zeros(self.world.batch_dim, device=self.world.device)

            return ((torch.stack([agent.rewards[r] for r in agent.rewards.keys()], dim=0).sum(dim=0)
                     * agent.state.reward_multiplier)
                    + task_reward
                    + self.final_rew)

    def reset_agent_rewards(self, agent):
        for reward in agent.rewards.keys():
            agent.rewards[reward][:] = 0

    def reset_final_rewards(self):
        self.final_pos_rew[:] = 0
        self.final_mix_rew[:] = 0
        self.final_rew[:] = 0

    def compute_final_rewards(self):
        # Calculate a reward based on the minimum number of times ALL agents have completed a goal.
        # completion_progress = torch.stack([a.state.reward_multiplier for a in self.agent_list['nav']])
        # min_completion = torch.min(completion_progress, dim=0)[0] - 1
        # self.final_rew += min_completion * self.per_goal_reward

        # Add a reward if all gaols have been reached TODO: If this actually helps, make an experimental parameter
        # self.final_rew += torch.all(self.completed_goals, dim=0) * 1

        # Calculate a reward for each completed goal.
        # self.final_rew += torch.sum(self.completed_goals, dim=0) * self.per_goal_reward

        if self.per_agent_reward:
            # Final reward is proportional to num agents who have reached their goal.
            for a in self.agent_list['nav']:
                self.final_pos_rew += a.rewards["final"]
            self.final_rew += self.final_pos_rew
        else:
            all_tasks_complete = torch.all(torch.stack(
                [a.state.task_complete for a in self.agent_list['nav']]
            ), dim=0)
            # Add final_pos_reward to batch dims when all agents have reached their goals.
            self.final_pos_rew[all_tasks_complete] += self.final_pos_reward
            # self.final_rew[all_tasks_complete] += self.final_pos_reward

        if self.per_agent_reward:
            # Final reward is proportional to num agents who have reached their goal.
            for a in self.agent_list['mix']:
                self.final_mix_rew += a.rewards["final"]
            self.final_rew += self.final_mix_rew
        else:
            all_tasks_complete = torch.all(torch.stack(
                [a.state.task_complete for a in self.agent_list['mix']]
            ), dim=0)
            # Add all_mixed reward to batch dims when all agents have mixed the correct solution.
            self.final_mix_rew[all_tasks_complete] += self.final_mix_reward
            # self.final_rew[all_tasks_complete] += self.final_mix_reward
        self.final_rew[torch.logical_and(self.final_pos_rew > 0, self.final_mix_rew > 0)] += self.final_pos_reward + self.final_mix_reward

    def compute_positional_rewards(self, agent):
        if agent.task != "mix":
            agent_index = int(agent.name.split("_")[1])
            dists = torch.stack(
                [
                    torch.linalg.vector_norm((agent.state.pos - goal.state.pos), dim=1)
                    for goal in self.goals
                ])

            if self.pos_shaping:
                # Perform position shaping on goal distances
                pos_shaping = dists * self.pos_shaping_factor

                shaped_dists = (agent.pos_shaping - pos_shaping) / agent.pos_shape_norm
                agent.pos_shaping = pos_shaping

                # TODO: This is sub-optimal for tensor operations.. find a better method
                target_shaped_dists = torch.stack([
                    shaped_dists[agent.state.target_goal[i], i] for i in range(len(agent.state.target_goal))
                ])
                target_shaped_dists = shaped_dists[agent_index]
                agent.rewards["position"] += target_shaped_dists

            # TODO: This is sub-optimal for tensor operations.. find a better method
            target_dists = torch.stack([
                dists[agent.state.target_goal[i], i] for i in range(len(agent.state.target_goal))
            ])
            target_dists = dists[agent_index]
            # TODO: Non-multi-head implementations will overwrite this state when computing mixing success..
            agent.state.task_complete = target_dists <= agent.shape.radius
            agent.rewards["final"][agent.state.task_complete] += \
                (self.final_pos_reward / self.n_agents) if self.final_pos_reward > 0 else self.default_task_rew

    def compute_mixing_rewards(self, agent):
        if agent.task != "nav":
            agent_index = int(agent.name.split("_")[1])
            knowledge_dists = torch.stack(
                [
                    torch.linalg.vector_norm(
                        (agent.state.knowledge[:, 1, :] - goal.state.expected_knowledge), dim=1)
                    for goal in self.goals
                ])

            if self.mix_shaping:
                # Perform reward shaping on knowledge mixtures.
                mix_shaping = knowledge_dists * self.mix_shaping_factor
                shaped_mixes = (agent.mix_shaping - mix_shaping) / agent.mix_shaping_norm
                agent.mix_shaping = mix_shaping

                # TODO: Sub-optimal computation
                target_shaped_mixes = torch.stack([
                    shaped_mixes[agent.state.target_goal[i], i] for i in range(len(agent.state.target_goal))
                ])
                target_shaped_mixes = shaped_mixes[agent_index]
                agent.rewards["mixing"] += target_shaped_mixes

            # Dist below threshold
            # TODO: Sub-optimal computation
            target_knowledge_dists = torch.stack([
                knowledge_dists[agent.state.target_goal[i], i] for i in range(len(agent.state.target_goal))
            ])
            target_knowledge_dists = knowledge_dists[agent_index]
            # TODO: Do we need a logical OR here?
            agent.state.task_complete = target_knowledge_dists <= self.mixing_thresh
            agent.rewards["final"][agent.state.task_complete] += \
                (self.final_mix_reward / self.n_agents) if self.final_mix_reward > 0 else self.default_task_rew

    def compute_collision_penalties(self, agent):
        if agent.task != "mix":
            if self.agent_collision_penalty != 0:
                for a in self.agent_list['nav']:
                    if a != agent:
                        agent.rewards["agent_collision"][
                            self.world.get_distance(agent, a) <= self.min_collision_distance
                            ] += self.agent_collision_penalty
            if self.env_collision_penalty != 0:
                for l in self.world.landmarks:
                    if self.world.collides(agent, l):
                        if l in self.world.walls:
                            penalty = self.env_collision_penalty
                        else:
                            penalty = 0
                        agent.rewards["obstacle_collision"][
                            self.world.get_distance(agent, l) <= self.min_collision_distance
                            ] += penalty

    # --------- Action Pre-processing -------- #

    def process_action(self, agent: DOTSAgent):
        if agent.task != "nav" and agent in self.agent_list["mix"]:
            if self.multi_head:
                # Update the mix agent position to be equal to the position of the nav agent counterpart.
                agent_index = int(agent.name.split('_')[-1])
                agent.state.pos = self.agent_list["nav"][agent_index].state.pos
            self.mix_knowledge(agent)

    def set_new_goal(self, agent):
        if agent.task != "mix":
            if self.multi_head:
                # Get the counter-part agent and check both have been assigned rewards (i.e. on goal and mixed)
                counter_part = self.agent_list["mix"][agent.agent_index]
                # Get the batch dim where agents have completed both mixing and positional tasks for a goal.
                update_goal = torch.logical_and(agent.state.task_complete, counter_part.state.task_complete)
                # Mark the agents target goal index, at the batch dim, as completed.
                self.completed_goals[agent.state.target_goal[update_goal], update_goal] = True
            else:
                # TODO: This isn't useful: we need to find a better way to compute completion
                #  for non-multi-head instances.
                update_goal = agent.rewards["final"] > 0.03
                self.completed_goals[agent.state.target_goal[update_goal], update_goal] = True

            if torch.any(update_goal):
                # Assign goals sequentially, therefore next index is sum of goal selected so far.
                # NOTE: Could be part of the training challenge as the network can't predict
                #  which goals are likely to be selected..
                next_goal_index = torch.sum(self.selected_goals, dim=0)
                # We do not want to update the target goal if we have already assigned all goals.
                update_goal[next_goal_index >= self.n_goals] = False
                self.selected_goals[next_goal_index[update_goal], update_goal] = True

                if torch.any(update_goal):
                    agents_to_update = [agent, counter_part] if self.multi_head else [agent]
                    for a in agents_to_update:
                        # Increase the multiplier, update the agents target, and reset task_complete to False.
                        a.state.reward_multiplier[update_goal] *= 2

                        # Ensuring int conversion as there is some outstanding bug causing
                        # next_goal_index to be a long after 50+iterations.?
                        a.state.target_goal[update_goal] = next_goal_index[update_goal].int()
                        a.state.task_complete[update_goal] = False

    def mix_knowledge(self, agent: DOTSAgent):
        # TODO: Allow re-mixing after correct mix has been produced. Ie. remove constraint to ensure proper learning.
        """
        Uses the agent coms signal (c[0]) to determine if mix is desired.
        Uses the other agent coms signal (c[1:]) to read the communicated knowledge.
        Args:
            agent: The agent requesting the knowledge mix.
        """
        agent_index = int(agent.name.split('_')[-1])

        # Establish where a request is true along the batch dim,
        #  and the agent hasn't already produced the correct solution.
        request_mix = torch.logical_and(agent.action.c[:, 0] > 0.5, ~agent.state.task_complete.squeeze())
        if not self.isolated_coms:
            #   Copy for (n_agents - 1) to simplify tensor calcs.
            request_mix = request_mix.unsqueeze(0).repeat(self.n_agents - 1, 1)

        # Find all agents in proximity along the batch dim. Shape: [n_agents-1, batch dim]
        #  Multiply by request_mix to zero any batch dim where the agent is not requesting to mix.
        in_prox = (((torch.stack(
            [torch.linalg.norm(agent.state.pos - other.state.pos, dim=1)
             for other in self.agent_list['mix'] if other != agent]))
                    < self.coms_proximity) * request_mix)
        any_in_prox = torch.any(in_prox, dim=0)

        # Clone existing agent learnt knowledge.
        new_mix = agent.state.knowledge[:, 1, :].clone()

        # Collect agent mixing weights across batch dim.
        if self.learn_mix:
            # Mixing weights should be the last n values of the action vector, where n is the dimension of the knowledge.
            knowledge_dims = self.knowledge_shape[-1]
            # # NOTE: DEBUGGING Interactive Render
            # agent.action.u[:, -knowledge_dims:] = (self.goals[agent.state.target_goal].state.expected_knowledge * 2) - 1
            mix_coefficients = (agent.action.u[:, -knowledge_dims:] + 1) / 2  # Shift from [-1, 1] to [0,1]
        else:
            mix_coefficients = self.goals[agent.state.target_goal].state.expected_knowledge

        # Collect incoming coms signals across the batch dim.
        if self.learn_coms:
            if self.isolated_coms:
                coms_index = agent_index * self.knowledge_shape[1]
                # Get the relevant communication output from the coms GNN
                com_knowledge = self.coms_network.action.u[:, coms_index: coms_index + self.knowledge_shape[1]]
                com_knowledge = (com_knowledge + 1) / 2  # Shift from [-1, 1] to [0,1]
            else:
                # Get the communicated knowledge from other agents coms signals
                com_knowledge = torch.stack(
                    [other.state.c[:, 1:] for other in self.agent_list['mix'] if other != agent]
                )
        else:
            # If not learning to communicate, ignore coms signals and just take collect source knowledge.
            com_knowledge = torch.stack(
                [other.state.knowledge[:, 0, :] for other in self.agent_list['mix'] if other != agent]
            )

        if self.isolated_coms:
            new_mix = com_knowledge * mix_coefficients

        else:
            # Zero all learnt knowledge vectors which will be updated.
            for i in range(in_prox.shape[0]):
                new_mix[in_prox[i], :] = 0

            # Update the agent learnt knowledge with the weighted mix of the others' communicated knowledge.
            for i in range(in_prox.shape[0]):
                new_mix[in_prox[i], :] += com_knowledge[i, in_prox[i], :] * mix_coefficients[in_prox[i], :]

            # Add agents own source knowledge along any dim where a mix has been made.
            new_mix[any_in_prox, :] += agent.state.knowledge[any_in_prox, 0, :] * mix_coefficients[any_in_prox, :]

        # TODO: Remove - used for debugging coms signals.
        if not self.isolated_coms:
            # Sum incoming communications across the agent stack dimension.
            agent.incoming_coms = torch.sum(com_knowledge, dim=0)
            # Add own knowledge to coms
            agent.incoming_coms += agent.state.knowledge[:, 0, :]
        else:
            agent.incoming_coms = com_knowledge
        # self.debug_coms_signals(agent, com_knowledge, any_in_prox, in_prox, new_mix)
        agent.state.knowledge[:, 1, :] = new_mix

    # ---------- Completion Signals ---------- #

    def done(self):
        return torch.all(self.completed_goals, dim=0)

    def info(self, agent: DOTSAgent) -> dict:
        if type(agent).__name__ == "DOTSAgent":
            if agent.task == "mix":
                return {
                    "agent_shaping": agent.rewards["mixing"],
                    "agent_final": agent.rewards["final"],
                    "final_reward": self.final_rew
                }

            elif agent.task == "nav":
                return {
                    "agent_collision": agent.rewards["agent_collision"],
                    "obstacle_collision": agent.rewards["obstacle_collision"],
                    "agent_shaping": agent.rewards["position"],
                    "agent_final": agent.rewards["final"],
                    "final_reward": self.final_rew
                }

            else:
                return {
                    "agent_collision": agent.rewards["agent_collision"],
                    "obstacle_collision": agent.rewards["obstacle_collision"],
                    "pos_shaping": agent.rewards["position"],
                    "mix_shaping": agent.rewards["mixing"],
                    "agent_final": agent.rewards["final"],
                    "final_reward": self.final_rew
                }
        elif type(agent).__name__ == "DOTSComsNetwork":
            return {
                "final_rew": self.final_rew
            }

    # ------------  Rendering --------------- #

    def top_layer_render(self, env_index: int = 0):
        geoms = []

        # Render landmark knowledge
        for landmark in self.world.landmarks:
            if isinstance(landmark, DOTSPayloadDest):
                goal_index = int(landmark.name.split('_')[-1])
                if self.completed_goals[goal_index, env_index]:
                    base_l, base_r, base_t, base_b = (
                        0, landmark.shape.width, landmark.shape.length / 2, -landmark.shape.length / 2
                    )
                    base = rendering.make_polygon(
                        [(base_l, base_b), (base_l, base_t), (base_r, base_t), (base_r, base_b)])
                    base.set_color(1, 1, 0)
                    base_xform = rendering.Transform()
                    base_xform.set_translation(landmark.state.pos[env_index][X] - base_r / 2,
                                               landmark.state.pos[env_index][Y])
                    base.add_attr(base_xform)
                    geoms.append(base)

                l, r, t, b = 0, landmark.shape.width / 2, landmark.shape.length / 4, -landmark.shape.length / 4
                knowledge = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
                col = landmark.state.expected_knowledge[env_index].reshape(-1)[:3]
                knowledge.set_color(*col)
                xform = rendering.Transform()
                xform.set_translation(landmark.state.pos[env_index][X] - r / 2, landmark.state.pos[env_index][Y])
                knowledge.add_attr(xform)
                geoms.append(knowledge)

        # Re-render agents (on top of landmarks) and agent knowledge.
        for i, agent in enumerate(self.agent_list['nav']):
            # test = agent.render(env_index=env_index)
            for elem in agent.render(env_index=env_index):
                geoms.append(elem)
            primary_knowledge = rendering.make_circle(proportion=0.5, radius=self.agent_radius / 2)
            mixed_knowledge = rendering.make_circle(proportion=0.5, radius=self.agent_radius / 2)

            mix_head = self.agent_list['mix'][i]
            primary_col = mix_head.state.knowledge[env_index][0].reshape(-1)[:3]
            mixed_col = mix_head.state.knowledge[env_index][1].reshape(-1)[:3]

            # Add a yellow ring around agents who have successfully matched their knowledge.
            if mix_head.state.task_complete[env_index]:
                success_ring = rendering.make_circle(radius=self.agent_radius, filled=True)
                success_ring.set_color(1, 1, 0)
                s_xform = rendering.Transform()
                s_xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
                success_ring.add_attr(s_xform)
                geoms.append(success_ring)

            primary_knowledge.set_color(*primary_col)
            mixed_knowledge.set_color(*mixed_col)

            p_xform = rendering.Transform()
            primary_knowledge.add_attr(p_xform)

            m_xform = rendering.Transform()
            mixed_knowledge.add_attr(m_xform)

            p_xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
            p_xform.set_rotation(math.pi / 2)
            m_xform.set_translation(agent.state.pos[env_index][X], agent.state.pos[env_index][Y])
            m_xform.set_rotation(-math.pi / 2)

            # label = TextLine(f"Agent {i}", x=agent.state.pos[env_index][X], y=agent.state.pos[env_index][Y] - 10)
            # geoms.append(label)

            geoms.append(primary_knowledge)
            geoms.append(mixed_knowledge)

            # TODO: Using for debugging, one or other to avoid overlapping text.
        for agent in self.all_agents:
            self.render_rewards(agent, env_index, geoms)
            if agent.task == "mix":
                self.render_mix_actions(agent, env_index, geoms)
            if agent.task == "nav":
                self.render_nav_actions(agent, env_index, geoms)

        # TODO: Add agent labels?

        return geoms

    def render_mix_actions(self, agent, env_index, geoms):
        if agent.action.u is not None:
            incoming_coms = ("coms_in: ["
                             + ",".join([f"{c:.2f}" for c in agent.incoming_coms[env_index]])
                             + "]")

            vert_offset = 650 - (30 * self.n_agents) - 100 - (15 * self.n_agents)
            coms_line = rendering.TextLine(f'{agent.name} {incoming_coms}',
                                           y=(vert_offset - (15 * (int(agent.name.split('_')[-1]) + 1))),
                                           font_size=10)

            outgoing_coms = ("coms_out: ["
                             + ",".join([f"{c:.2f}" for c in agent.state.c[env_index]])
                             + "]  source: ["
                             + ",".join([f"{p:.2f}" for p in agent.state.knowledge[env_index, 0, :]])
                             + "]"
                             )
            vert_offset -= ((15 * self.n_agents) + 10)
            out_coms_line = rendering.TextLine(f'{agent.name} {outgoing_coms}',
                                               y=(vert_offset - (15 * (int(agent.name.split('_')[-1]) + 1))),
                                               font_size=10)

            action = ("mix_coef: ["
                      + ",".join([f"{(a + 1) / 2:.2f}" for a in agent.action.u[env_index][-self.knowledge_shape[-1]:]])
                      + "]")
            vert_offset -= ((15 * self.n_agents) + 10)
            action_line = rendering.TextLine(f'{agent.name} {action}',
                                             y=(vert_offset - (15 * (int(agent.name.split('_')[-1]) + 1))),
                                             font_size=10)

            target_goal = agent.state.target_goal[env_index]
            knowledge = ("learnt: ["
                         + ",".join([f"{p:.2f}" for p in agent.state.knowledge[env_index, 1, :]])
                         + "]  goal: ["
                         + ",".join(
                        [f"{g:.2f}" for g in self.goals[target_goal].state.expected_knowledge[env_index]])
                         + "]"
                         )
            vert_offset -= ((15 * self.n_agents) + 10)
            knowledge_line = rendering.TextLine(f'{agent.name} {knowledge}',
                                                y=(vert_offset - (15 * (int(agent.name.split('_')[-1]) + 1))),
                                                font_size=10)

            vert_offset -= ((15 * self.n_agents) + 30)

            s_goals = ("Selected Goals: ["
                       + ",".join([f"{val}" for val in self.selected_goals[:, env_index]])
                       + "]")
            c_goals = ("Completed Goals: ["
                       + ",".join([f"{val}" for val in self.completed_goals[:, env_index]])
                       + "]")
            selected_goals = rendering.TextLine(s_goals, y=vert_offset, font_size=10)
            completed_goals = rendering.TextLine(c_goals, y=vert_offset - 15, font_size=10)

            geoms.append(coms_line)
            geoms.append(out_coms_line)
            geoms.append(action_line)
            geoms.append(knowledge_line)
            geoms.append(selected_goals)
            geoms.append(completed_goals)

    def render_nav_actions(self, agent, env_index, geoms):
        if agent.action.u is not None:
            action = ("vel: ["
                      + ",".join([f"{a:.2f}" for a in agent.action.u[env_index][:2]])
                      + "]")

            vert_offset = 650 - (30 * self.n_agents) - 20
            target_goal = rendering.TextLine(f'{agent.name} target: {agent.state.target_goal[env_index]}',
                                             y=(vert_offset - (15 * (int(agent.name.split('_')[-1]) + 1))),
                                             font_size=10)

            vert_offset -= ((15 * self.n_agents) + 20)
            action_line = rendering.TextLine(f'{agent.name} {action}',
                                             y=(vert_offset - (15 * (int(agent.name.split('_')[-1]) + 1))),
                                             font_size=10)

            geoms.append(target_goal)
            geoms.append(action_line)

    def render_rewards(self, agent, env_index, geoms):
        rew_components = torch.stack([agent.rewards[r][env_index] for r in agent.rewards.keys()], dim=0)
        total_rew = rew_components.sum(dim=0) * agent.state.reward_multiplier[env_index]
        init_pos = 665 if "nav" in agent.task else 650
        reward = rendering.TextLine(f'{agent.name} rew: '
                                    + ",".join(
            [f'{r * agent.state.reward_multiplier[env_index]:.2f}' for r in rew_components])
                                    + f' = {total_rew:.2f}', x=0,
                                    y=(init_pos - (30 * (int(agent.name.split('_')[-1]) + 1))),
                                    font_size=10)
        geoms.append(reward)

        if agent == self.world.agents[-1]:
            final_rew = rendering.TextLine(f'final rew: {self.final_rew[env_index]:.2f}',
                                           y=650, font_size=10)
            geoms.append(final_rew)


if __name__ == '__main__':
    render_interactively(
        __file__,
        n_agents=3,
        n_goals=3,
        pos_shaping=True,
        mix_shaping=True,
        task_type="full",
        knowledge_shape=(2, 3),
        clamp_actions=True,
        agent_collision_penalty=-0.2,
        env_collision_penalty=-0.2,
        final_mix_reward=0.1,
        final_pos_reward=0.1,
        multi_head=True,
        random_knowledge=False,
        debug_goals=False
    )