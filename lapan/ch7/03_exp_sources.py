import gym
import ptan
from typing import List, Optional, Tuple, Any


class ToyEnv(gym.Env):
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=3)
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return self.step_index

    def step(self, action):
        is_done = self.step_index == 10
        if is_done:
            return self.step_index % self.observation_space.n, 0.0, is_done, {}
        self.step_index += 1
        return self.step_index % self.observation_space.n, float(action), self.step_index == 10, {}


class DullAgent(ptan.agent.BaseAgent):
    def __init__(self, action):
        super(DullAgent, self).__init__()
        self.action = action

    def __call__(self, observations, state=None):
        return [self.action for _ in observations], state


if __name__ == '__main__':
    env = ToyEnv()
    agent = DullAgent(action=1)
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 15:
            break
        #print(exp)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env=env, agent=agent, gamma=1.0, steps_count=1)
    for idx, exp in enumerate(exp_source):
        print(exp)
        if idx > 10: break
