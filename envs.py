import gymnasium as gym
import numpy as np

def getEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Supported only continuous action spaces for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()

    return observationShape, actionSize, actionLow, actionHigh

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))/255.0
    

class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        observation, info = self.env.reset(seed=seed)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done