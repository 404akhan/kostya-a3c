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
import time
from collections import deque

def is_dead(info):
    dead = False
    if is_dead.current_life > info['ale.lives']:
        dead = True
    is_dead.current_life = info['ale.lives']
    return dead

is_dead.current_life = 0

def test(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space, args.num_skips)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    dead = False
    action_stat = [0] * (model.n_real_acts + model.n_aux_acts)

    start_time = time.time()
    episode_length = 0

    for ep_counter in itertools.count(1):
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            if not os.path.exists('model-a3c-aux'):
                os.makedirs('model-a3c-aux')
            torch.save(shared_model.state_dict(), 'model-a3c-aux/model-{}.pth'.format(args.model_name))
            print('saved model')
        if done or dead:
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model(
            (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()

        action_np = action[0, 0]
        action_stat[action_np] += 1

        if action_np < model.n_real_acts:
            state, reward, done, info = env.step(action_np)
            dead = is_dead(info)
            done = done or episode_length >= args.max_episode_length
            reward_sum += reward
            episode_length += 1

            if args.testing:
                env.render()
                if done or dead:
                    cx = Variable(torch.zeros(1, 256), volatile=True)
                    hx = Variable(torch.zeros(1, 256), volatile=True)
                value, logit, _ = model(
                    (Variable(torch.from_numpy(state).unsqueeze(0)), (hx, cx)))
                print('episode', episode_length, 'normal action', action_np, 'lives', info['ale.lives'], \
                    'rew', reward, 'value', value.data.numpy()[0][0], '!!!' if reward !=0 else '')
        else:
            for counter_skips in range(action_np - model.n_real_acts + 2):
                state, rew, done, info = env.step(0)  # instead of random perform NOOP=0
                dead = is_dead(info)
                done = done or episode_length >= args.max_episode_length

                reward_sum += rew
                episode_length += 1

                if args.testing:
                    env.render()
                    if done or dead:
                        cx = Variable(torch.zeros(1, 256), volatile=True)
                        hx = Variable(torch.zeros(1, 256), volatile=True)
                    value, logit, _ = model(
                        (Variable(torch.from_numpy(state).unsqueeze(0)), (hx, cx)))
                    print('episode', episode_length, 'random action', action_np, 'lives', info['ale.lives'], \
                        'rew', rew, 'value', value.data.numpy()[0][0], '!!!' if rew !=0 else '')

                if done or dead:
                    break
                if counter_skips != action_np - model.n_real_acts + 1: # maintain hx, cx for conseq frames
                    _, _, (hx, cx) = model((Variable(torch.from_numpy(state).unsqueeze(0)), (hx, cx)))

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            print("actions stats real {}, aux {}".format(action_stat[:model.n_real_acts], action_stat[model.n_real_acts:]))

            reward_sum = 0
            episode_length = 0
            state = env.reset()
            action_stat = [0] * (model.n_real_acts + model.n_aux_acts)
            if not args.testing: time.sleep(60)
            env.seed(args.seed + rank + (args.num_processes+1)*ep_counter)

        state = torch.from_numpy(state)
