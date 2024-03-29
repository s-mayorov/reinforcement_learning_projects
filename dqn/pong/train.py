import argparse
import collections
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from wrappers import make_env
from gym_experiments.dqn.model import DQN
from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
EPSILON_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, new_states = batch
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    new_states_v = torch.tensor(new_states).to(device)
    dones_v = torch.BoolTensor(dones)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(new_states_v).max(1)[0]
    next_state_values[dones_v] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help=f"Name of the environment, default={DEFAULT_ENV_NAME}")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    env = make_env(args.env)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-"+args.env)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = 0

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START-frame_idx/EPSILON_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(f"{frame_idx} done: {len(total_rewards)} games, reward {m_reward:.3f}, eps {epsilon:.2f}, speed {speed:.2f}")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward is None  or best_m_reward < m_reward:
                torch.save(net.state_dict(), f"pong_{int(m_reward)}.pt")
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.2f} -> {m_reward:.2f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print(f"Solved in {frame_idx} frames!")
                break

        if len(buffer) < REPLAY_START_SIZE: continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()







