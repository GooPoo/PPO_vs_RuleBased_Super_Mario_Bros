import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
import numpy as np


# Requirements
# gym 0.17.3
# pyTorch 1.10.1
# CPU   pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
# GPU - CUDA 11.1 REQUIRED   pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# stable-baselines3 1.3.0 


def run_multiple_episodes(model_path, num_episodes):
    model = PPO.load(model_path)
    env = gym_super_mario_bros.make('SuperMarioBros-1-2-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    episode_info = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        total_actions = 0
        done = False
        furthest_x = 0

        while not done:
            action, _ = model.predict(state)
            state, reward, done, info_list = env.step(action)
            env.render()

            for info in info_list:
                mario_world_x = info["x_pos"]
                time = info["time"]
                if furthest_x < mario_world_x:
                    furthest_x = mario_world_x
                if time <= 1:
                    print("works")
                    done = True

            total_reward += reward
            total_actions += 1


        episode_info.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'furthest_x': furthest_x,
            'total_actions': total_actions
        })

    avg_reward = np.mean([info['total_reward'] for info in episode_info])
    avg_furthest_x = np.mean([info['furthest_x'] for info in episode_info])
    avg_total_actions = np.mean([info['total_actions'] for info in episode_info])

    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    print(f"Average Furthest x reached over {num_episodes} episodes: {avg_furthest_x}")
    print(f"Average actions over {num_episodes} episodes: {avg_total_actions}")

    return episode_info


# Models have been removed for the repo, must run stablePPO.py to create models to run.
# Once you have models, move them into this directory and change model_path to the model.
if __name__ == "__main__":
    model_path = "C:/Users/leebe\Desktop/uni/CITS3001-Algorithms/Project-Mario/PPO-Model/#"
    num_episodes = 5
    episode_info = run_multiple_episodes(model_path, num_episodes)