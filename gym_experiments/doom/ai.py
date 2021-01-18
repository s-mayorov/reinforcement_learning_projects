import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import vizdoomgym


class CNN(nn.Module):
    def __init__(self, n_actions, input_shape):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class SoftmaxBody:
    def __init__(self, T):
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions


class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, images, *args, **kwargs):
        inputs = torch.tensor(np.array(images, dtype=np.float32))
        outputs = self.brain(inputs)
        actions = self.body.forward(outputs)
        return actions.data.numpy()








# if __name__ == '__main__':

#
#     env = gym.make('VizdoomBasic-v0')
#
#     # use like a normal Gym environment
#     state = env.reset()
#     for _ in range(100):
#         state, reward, done, info = env.step(env.action_space.sample())
#         env.render()
#     env.close()