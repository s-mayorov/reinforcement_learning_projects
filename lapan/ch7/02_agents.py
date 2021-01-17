import ptan
import torch
import torch.nn as nn


class DQNNet(nn.Module):
    def __init__(self, actions):
        super(DQNNet, self).__init__()
        self.actions = actions

    def forward(self, x):
        return torch.eye(x.size()[0], self.actions)


class PolicyNet(nn.Module):
    def __init__(self, actions):
        super(PolicyNet, self).__init__()
        self.actions = actions

    def forward(self, x):
        shape = (x.size()[0], self.actions)
        res = torch.zeros(shape, dtype=torch.float32)
        res[:, 0] = 1.0
        res[:, 1] = 1.0
        return res


if __name__ == '__main__':
    net = DQNNet(actions=3)
    print(net(torch.zeros(3, 10)))

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)
    print(agent(torch.zeros(2, 5)))

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)
    print(agent(torch.zeros(10, 5))[0])

    net = PolicyNet(actions=5)
    print(net(torch.zeros(6, 10)))
    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True)
    print(agent(torch.zeros(6, 5))[0])
