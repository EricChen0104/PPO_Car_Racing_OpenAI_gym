import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # 確保批次數量適應資料量
        batches = [indices[i:i + self.batch_size] for i in batch_start]


        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.actions.append(action)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class DiagonalGaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_mean = nn.Linear(in_dim, out_dim)
        self.b_logstd = nn.Parameter(T.zeros(1, out_dim))

    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = T.zeros_like(mean) + self.b_logstd

        return T.distributions.Normal(mean, logstd.exp())

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=64, fc2_dims=64, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.conv = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with T.no_grad():
            conv_out = np.prod(self.conv(T.zeros(1, *input_dims)).size()).item()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(fc2_dims, fc2_dims),
            nn.ReLU(),
            DiagonalGaussian(fc2_dims, n_actions),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.conv(state)
        x = self.fc(x)
        dist = self.actor_head(x)
        return dist 

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=64, fc2_dims=64,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        self.conv = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with T.no_grad():
            conv_out = np.prod(self.conv(T.zeros(1, *input_dims)).size()).item()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(fc2_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.conv(state)
        state = self.fc(state)
        value = self.critic_head(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = dist.log_prob(action).squeeze(0).detach().cpu().numpy()  # 形狀 (3,)
        action = action.squeeze(0).detach().cpu().numpy()  # 形狀 (3,)
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        if len(self.memory.states) < self.memory.batch_size:
            print("Not enough data in memory to learn. Skipping...")
            return

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs.sum(-1).exp() / old_probs.sum(-1).exp())

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                entropy = dist.entropy().sum(-1).mean() 

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()