import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim
import gymnasium as gym
import numpy as np
import ale_py

"link to code"
"https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium?dc_referrer=https%3A%2F%2Fwww.google.com%2F"

#env = gym.make("ALE/Freeway-v5", obs_type="grayscale",render_mode = "human")
env = gym.make("ALE/Freeway-v5", obs_type="grayscale")
#env = gym.make("CartPole-v1")


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = self.layer1(x)
        #print(f"After layer1: {x.shape}")
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x
    
def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    normalized_returns = (returns - returns.mean()) / returns.std()
    return normalized_returns

def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0
    policy.train()
    observation, info = env.reset()
    while not done:
        observation = observation.flatten()
        #print("observation shape: ", observation.shape)
        observation = torch.FloatTensor(observation).unsqueeze(0)
        #print("observation shape: ", observation.shape)
        action_pred = policy(observation)
        action_prob = F.softmax(action_pred, dim = -1)
        #print(f"Action prob: {action_prob}")
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward
    log_prob_actions = torch.cat(log_prob_actions)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)
    return episode_return, stepwise_returns, log_prob_actions

def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss

def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main(): 
    MAX_EPOCHS = 10
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 1
    INPUT_DIM = env.observation_space.shape[0]*env.observation_space.shape[1]
    print(f"Input dim: {INPUT_DIM}")
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n
    print(f"Output dim: {OUTPUT_DIM}")
    DROPOUT = 0.5
    episode_returns = []
    policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
    LEARNING_RATE = 0.01
    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)
    for episode in range(1, MAX_EPOCHS+1):
        episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, DISCOUNT_FACTOR)
        _ = update_policy(stepwise_returns, log_prob_actions, optimizer)
        episode_returns.append(episode_return)
        #print(episode_returns)
        mean_episode_return = np.mean(episode_returns[-N_TRIALS:])
        if episode % PRINT_INTERVAL == 0:
            print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')
        if mean_episode_return >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

main()