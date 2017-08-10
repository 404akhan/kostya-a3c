import math
import os
import sys
import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def is_dead(info):
    dead = False
    if is_dead.current_life > info['ale.lives']:
        dead = True
    is_dead.current_life = info['ale.lives']
    return dead

is_dead.current_life = 0

def train(rank, args, shared_model, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space, args.num_skips)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    dead = False

    episode_length = 0
    for ep_counter in itertools.count(1):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done or dead:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            action_np = action.numpy()[0][0]
            if action_np < model.n_real_acts:
                state, reward, done, info = env.step(action_np)
                dead = is_dead(info)
                done = done or episode_length >= args.max_episode_length
                reward = max(min(reward, 1), -1)
                episode_length += 1
            else:
                reward = 0.
                for counter_skips in range(action_np - model.n_real_acts + 2):
                    state, rew, done, info = env.step(0)  # instead of random perform NOOP=0
                    dead = is_dead(info)
                    done = done or episode_length >= args.max_episode_length
                    rew = max(min(rew, 1), -1)

                    reward += rew
                    episode_length += 1
                    if done or dead:
                        break

                    if counter_skips != action_np - model.n_real_acts + 1: # maintain hx, cx for conseq frames
                        _, _, (hx, cx) = model((Variable(torch.from_numpy(state).unsqueeze(0)), (hx, cx)))


            if done:
                episode_length = 0
                state = env.reset()
                env.seed(args.seed + rank + (args.num_processes+1)*ep_counter)

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done or dead:
                break

        R = torch.zeros(1, 1)
        if not done and not dead:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
