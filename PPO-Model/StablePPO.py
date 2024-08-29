import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import os 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
#from gym import Wrapper

# Requirements
# gym 0.17.3
# pyTorch 1.10.1
# CPU   pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
# GPU - CUDA 11.3 REQUIRED   conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# stable-baselines3 1.3.0 

# GPU is needed over CPU, leads to faster learning.


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

state = env.reset()
state, reward, done, info = env.step([5])

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            steps_on_loaded_model = 9000000
            totalSteps = steps_on_loaded_model + self.n_calls
            model_path = os.path.join(self.save_path, 'model_1{}'.format(totalSteps))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
model.learn(total_timesteps=5000000, callback=callback)
model.save('Final_model')
model = PPO.load('Final_model')


'''
# learning off previous model
model = PPO.load("best_model_9000000")
model.set_env(env)
model.learn(total_timesteps=1000000, callback=callback)
model.save('Final_model')
model = PPO.load('Final_model')
'''

state = env.reset()
while True: 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()