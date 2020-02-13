import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pdb

parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with DDPG')
parser.add_argument(
    '--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.9)')

parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

episodes = 5000
episode_length = 1000
render_interval = 500
memory_size = 10*episode_length

h=512
intermediate_layer=True

torch.manual_seed(args.seed)
np.random.seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

# task = 'pendulum'
task = 'walker'

if task == 'pendulum':
    env = gym.make('Pendulum-v0')
    state_dim = 3
    action_dim = 1
elif task == 'walker':
    env = gym.make('Walker2d-v2')
    state_dim = 17
    action_dim = 6

class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(state_dim, h)
        self.l2 = nn.Linear(h,h)
        self.mu_head = nn.Linear(h, action_dim)

    def forward(self, s):
        x = F.relu(self.fc(s))
        if intermediate_layer:
            x = F.dropout(F.relu(F.dropout(self.l2(x), p=.1)), p=.1)
        # u = 2.0 * F.tanh(self.mu_head(x))
        u = F.tanh(self.mu_head(x))
        return u


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(state_dim+action_dim, h)
        self.l2 = nn.Linear(h,h)
        self.v_head = nn.Linear(h, 1)

    def forward(self, s, a):
        x = F.relu(self.fc(torch.cat([s, a], dim=1)))
        if intermediate_layer:
            x = F.dropout(F.relu(F.dropout(self.l2(x), p=.1)), p=.1)
        state_value = self.v_head(x)
        return state_value

class TransitionNet(nn.Module):
    def __init__(self):
        super(TransitionNet, self).__init__()
        self.fc = nn.Linear(state_dim+action_dim, h)
        self.l2 = nn.Linear(h,h)
        self.v_head = nn.Linear(h, state_dim)

    def forward(self, s, a):
        x = F.relu(self.fc(torch.cat([s, a], dim=1)))
        if intermediate_layer:
            x = F.dropout(F.relu(F.dropout(self.l2(x), p=.1)), p=.1)
        state = self.v_head(x)
        return state

class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class Agent():

    max_grad_norm = 0.5

    def __init__(self):
        self.training_step = 0
        # self.var = 1.
        self.var = 1.
        self.eval_cnet, self.target_cnet = CriticNet().float().cuda(), CriticNet().float().cuda()
        self.eval_anet, self.target_anet = ActorNet().float().cuda(), ActorNet().float().cuda()
        self.memory = Memory(memory_size)
        self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-3)
        self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=3e-4)

        self.trans_model = TransitionNet().float()
        self.optimizer_t = optim.Adam(self.trans_model.parameters(), lr=1e-3)

    def select_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        state = torch.from_numpy(state).float().cuda()
        mu = self.eval_anet(state)
        dist = Normal(mu, torch.tensor(self.var, dtype=torch.float).cuda())
        action = dist.sample()
        # action.clamp(-2.0, 2.0)
        action.clamp(-1.0, 1.0)
        return action.cpu().numpy()
        # return (action.item(),)

    def save_param(self):
        torch.save(self.trans_model.state_dict(), 'param/ddpg_transmodel_params.pkl')
        torch.save(self.eval_anet.state_dict(), 'param/ddpg_anet_params.pkl')
        torch.save(self.eval_cnet.state_dict(), 'param/ddpg_cnet_params.pkl')

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float).cuda()
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, action_dim).cuda()
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1).cuda()
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float).cuda()

        with torch.no_grad():
            q_target = r + args.gamma * self.target_cnet(s_, self.target_anet(s_))
        q_eval = self.eval_cnet(s, a)

        # update critic net
        self.optimizer_c.zero_grad()
        c_loss = F.smooth_l1_loss(q_eval, q_target)
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()

        # update transition net
        # self.optimizer_t.zero_grad()
        # t_loss = F.mse_loss(self.trans_model(s, a), s_)
        # t_loss.backward()
        # self.optimizer_t.step()

        use_penalty = False
        grad_penalty = 0
        if use_penalty:
            epsilon = 10**(-9)
            new_state = self.trans_model(s, self.eval_anet(s))
            new_values = self.eval_cnet(new_state, self.eval_anet(new_state))
            grad_norms = [] 
            for val in new_values:
                self.optimizer_t.zero_grad()
                val.backward(retain_graph=True)
                norm = sum([torch.sum(p.grad**2) for p in self.trans_model.parameters()])**.5
                grad_norms.append(norm)
            grad_norms = torch.stack(grad_norms)
            grad_penalty = epsilon*grad_norms


        # update actor net
        self.optimizer_a.zero_grad()
        a_loss = -self.eval_cnet(s, self.eval_anet(s)-grad_penalty).mean()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        if self.training_step % 200 == 0:
            self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        if self.training_step % 201 == 0:
            self.target_anet.load_state_dict(self.eval_anet.state_dict())

        self.var = max(self.var * 0.999, 0.01)

        return q_eval.mean().item()


def main():
    env.seed(args.seed)

    agent = Agent()

    training_records = []
    # running_reward, running_q = -1000, 0
    running_reward, running_q = 0, 0
    for i_ep in range(episodes):
        score = 0
        state = env.reset()
        agent.var = .6/(i_ep+1)**.5 + .4

        for t in range(episode_length):
            if i_ep % render_interval == 0 and args.render:
                env.render()
            action = agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            # agent.store_transition(Transition(state, action, (reward + 8) / 8, state_))
            agent.store_transition(Transition(state, action, reward, state_))
            state = state_
            if agent.memory.isfull:
                q = agent.update()
                running_q = 0.99 * running_q + 0.01 * q

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % args.log_interval == 0:
            print('Step {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
                i_ep, running_reward, running_q))
        if i_ep % args.log_interval*50 == 0:
            agent.save_param()
            with open('log/ddpg_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)

    env.close()

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("img/ddpg.png")
    plt.show()


if __name__ == '__main__':
    main()
