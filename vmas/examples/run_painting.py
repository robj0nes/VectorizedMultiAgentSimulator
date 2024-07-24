import time

import torch

from vmas import make_env
from vmas.simulator.utils import save_video


def run_painting(
        render: bool = False,
        save_render: bool = False,
        num_envs: int = 32,
        n_steps: int = 100,
        random_action: bool = False,
        device: str = "cpu",
        scenario_name: str = "waterfall",
        n_agents: int = 3,
        n_goals: int = 3,
        continuous_actions: bool = True,
        visualize_render: bool = True,
):
    """Example function to use a vmas environment

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        n_agents (int): Number of agents
        scenario_name (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    dict_spaces = True  # Weather to return obs, rewards, and infos as dictionaries with agent names
    # (by default they are lists of len # of agents)
    model = torch.load('/Users/vd20433/FARSCOPE/FirstYearProject/Checkpoints/Full Task/pre-trained_3A-3G/checkpoints/pre-trained_checkpoint_6000000.pt')
    # Set up loss modules.
    losses = {}
    for group in ["nav_agents", "mix_agents"]:
        losses[group].load_state_dict(model[f"loss_{group}"])

    env = make_env(
        scenario=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        dict_spaces=dict_spaces,
        wrapper=None,
        seed=None,
        # Environment specific variables
        n_agents=n_agents,
        n_goals=n_goals,
        pos_shaping=True,
        mix_shaping=True,
        task_type="full",
        knowledge_shape=(2, 3),
        clamp_actions=True,
        agent_collision_penalty=-0.2,
        env_collision_penalty=-0.2,
        multi_head=True
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    policies = {}
    for agent in env.agents:
        # TODO: Setup GNN model for each agent.
        # TODO: Load checkpoint for agent.
        pass

    for _ in range(n_steps):
        step += 1
        print(f"Step {step}")

        actions = {}
        obs, rews, dones, info = env.step(actions)

        if render:
            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
                visualize_when_rgb=visualize_render,
            )
            if save_render:
                frame_list.append(frame)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)


if __name__ == '__main__':
    run_painting(
        scenario_name="painting",
        render=True,
        save_render=False,
        random_action=False,
        continuous_actions=True,
        n_steps=1000,
        n_agents=3,
        n_goals=6
    )
