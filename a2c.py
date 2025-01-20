import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
from collections import deque
import cv2
import torch.nn.functional as F


def preprocess_state(state):
    # Resize to 84x84 and normalize pixel values to [0, 1]
    resized_state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    normalized_state = resized_state / 255.0
    return np.expand_dims(normalized_state, axis=0)  # Add channel dimension (1 here because grayscale)

# Define the A2C model
class A2CModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A2CModel, self).__init__()
        # architecture:
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute the size of the flattened output after Conv layers (input size for actor and critic!)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            n_flatten = self.feature_extractor(dummy_input).shape[1]

        # actor channel, takes the flattened output as input and outputs a probability distribution for the actions
        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1) #prob distribution
        )
        # critic channel, takes the flattened output as input and outputs a single value
        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x) #extracts features (main network)
        action_probs = self.actor(features) #actor computes probabilities
        value = self.critic(features) #critic judges state
        return action_probs, value

# Compute discounted returns
def compute_returns(rewards, dones, gamma):
    returns = []
    G = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - done)
        returns.insert(0, G)
    return returns

# Training loop
def train_a2c(env, model, optimizer, num_steps=5, gamma=0.99):
    state, _ = env.reset()
    state = preprocess_state(state)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    log_probs = []
    values = []
    rewards = []
    dones = []

    for _ in range(num_steps):
        # Forward pass
        action_probs, value = model(state)
        action = torch.argmax(action_probs, dim=-1).item()

        # Step in the environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_state(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        #log probability for the selected action
        log_probs.append(torch.log(action_probs[0, action]))
        # state value
        values.append(value)
        rewards.append(reward)
        dones.append(done)

        state = next_state
        if done:
            state, _ = env.reset()
            state = preprocess_state(state)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # Compute returns and losses
    returns = compute_returns(rewards, dones, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values)
    log_probs = [log_prob.unsqueeze(0) for log_prob in log_probs]
    log_probs = torch.cat(log_probs)

    advantage = returns - values.detach()
    actor_loss = -(log_probs * advantage).mean()
    critic_loss = F.mse_loss(values, returns)
    loss = actor_loss + critic_loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Loss: {loss.item():.4f}")

# Main training script
def main():
    # Initialize environment and model
    env = gym.make("ALE/Freeway-v5", obs_type='grayscale')
    num_actions = env.action_space.n
    input_shape = (1, 84, 84)

    model = A2CModel(input_shape, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Track performance
    episode_rewards = deque(maxlen=100)

    # Train for a number of episodes
    for episode in range(1000):
        state, _ = env.reset()
        state = preprocess_state(state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        episode_reward = 0
        done = False

        while not done:
            action_probs, _ = model(state)
            action = torch.argmax(action_probs, dim=-1).item()


            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)

        # Train the model every 5 episodes
        if episode % 5 == 0:
            train_a2c(env, model, optimizer)

        # Print average reward every 10 episodes
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()