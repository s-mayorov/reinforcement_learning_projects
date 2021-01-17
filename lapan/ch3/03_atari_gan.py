import random
import argparse
import cv2
import numpy as np

import gym
import gym.spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from tensorboardX import SummaryWriter

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32
        )

    def observation(self, observation):
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [env.reset() for env in envs]
    env_gen = iter(lambda: random.choice(envs), None)
    while True:
        e = next(env_gen)
        obs, reward, done, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32)
            batch_np *= 2.0 / 255 - 1
            yield torch.tensor(batch_np)
            batch.clear()

        if done:
            e.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action='store_true', help="Enable cuda computation"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = [InputWrapper(gym.make(name)) for name in ("Breakout-v0", "AirRaid-v0", "Pong-v0")]
    input_shape = envs[0].observation_space.shape

    discr = Discriminator(input_shape=input_shape).to(device)
    gen = Generator(output_shape=input_shape).to(device)

    loss = nn.BCELoss()

    discr_optimizer = optim.Adam(discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    gen_optimizer = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    writer = SummaryWriter()

    discr_losses = []
    gen_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    for batch_v in iterate_batches(envs):
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)
        gen_output_v = gen(gen_input_v)

        discr_optimizer.zero_grad()
        discr_output_true_v = discr(batch_v)
        discr_output_fake_v = discr(gen_output_v.detach())
        discr_loss = loss(discr_output_true_v, true_labels_v) + loss(discr_output_fake_v, fake_labels_v)
        discr_loss.backward()
        discr_optimizer.step()
        discr_losses.append(discr_loss.item())

        gen_optimizer.zero_grad()
        disc_output_v = discr(gen_output_v)
        gen_loss_v = loss(disc_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, np.mean(gen_losses),
                     np.mean(discr_losses))
            writer.add_scalar(
                "gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar(
                "dis_loss", np.mean(discr_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(
                gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(
                batch_v.data[:64], normalize=True), iter_no)