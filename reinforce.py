import numpy as np
import gym
import time
import Policy
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env = gym.make('CartPole-v0')
print(env.observation_space)
print(env.action_space)

policy = Policy.Policy().to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
  scores = []
  scores_deque = deque(maxlen=100)

  for i_episode in range(1, n_episodes+1):
    saved_log_probs = []
    rewards = []
    state = env.reset()

    for i in range(max_t):
      action, log_prob = policy.act(state)
      saved_log_probs.append(log_prob)
      state, reward, done, _ = env.step(action)
      rewards.append(reward)
      if done:
        break
    
    scores.append(sum(rewards))
    scores_deque.append(sum(rewards))

    discounts = [gamma**1 for i in range(len(rewards) + 1)]
    R = sum([a*b for a, b in zip(discounts, rewards)])

    policy_loss = []
    for log_prob in saved_log_probs:
      policy_loss.append(-log_prob*R)
    policy_loss = torch.cat(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    torch.save(policy.state_dict(), 'checkpoint.pth')

    if i_episode % print_every == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    if np.mean(scores_deque)>=195.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
        break
  return scores

scores = reinforce()

plt.plot(np.arange(1, len(scores)+1), scores)
plt.xlabel('Episodes')
plt.xlabel('Avg Score')
plt.savefig('graph.jpg')
plt.show()

# policy.load_state_dict(torch.load('checkpoint.pth'))

# env = gym.make('CartPole-v0')

# state = env.reset()
# for t in range(10000):
#     action, _ = policy.act(state)
#     env.render()
#     time.sleep(0.01)
#     state, reward, done, _ = env.step(action)
#     if done:
#         break 

# env.close()