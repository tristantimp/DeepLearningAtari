# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 01:09:09 2025

@author: Gebruiker
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
import cv2
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('TkAgg')  # needed for visualization apparently


def calculate_shaped_reward(env, current_position, previous_position, reached_goal):
    progress_reward = 0.1 if current_position > previous_position else -0.1
    if current_position == previous_position:
        progress_reward -= 0.05
    goal_reward = 1.0 if reached_goal else 0.0
    total_reward = progress_reward + goal_reward
    return total_reward


def visualize_state(state):
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()

    if len(state.shape) == 4:  # (batch_size, channels, height, width)
        state = state[0]

    plt.imshow(state[0], cmap='gray')  # Only visualize the first channel
    plt.title("State")
    plt.axis('off')
    plt.show()


def print_action_dist(action_count, count_per_action, num_actions):
    for action in range(num_actions):
        print(f"Model performed ACTION {action} {count_per_action[action] / action_count}%")

        
def preprocess_state(state):
    resized_state = cv2.resize(state, (128, 128)) / 255.0  # Resize and normalize
    return torch.tensor(resized_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class A2CModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A2CModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            n_flatten = self.feature_extractor(dummy_input).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


def compute_returns(rewards, dones, gamma):
    returns = []
    G = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - done)
        returns.insert(0, G)
    return returns


def train_a2c(env, model, optimizer, replay_buffer, batch_size=64, num_steps=10, gamma=0.99):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    action_probs, values = model(states)
    next_action_probs, next_values = model(next_states)

    returns = rewards + gamma * next_values.squeeze() * (1 - dones)
    advantage = returns - values.squeeze()

    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = F.mse_loss(values.squeeze(), returns.detach())

    loss = actor_loss + critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Loss: {loss.item():.4f}")


def save_checkpoint(model, optimizer, filename="a2c_checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename="a2c_checkpoint.pth"):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {filename}")
        return True
    return False


import json  # Import for saving/loading progress

def save_rewards(reward_history, filename="reward_history.json"):
    """Save reward history to a JSON file."""
    with open(filename, "w") as f:
        json.dump(reward_history, f)
    print(f"Reward history saved to {filename}")


def load_rewards(filename="reward_history.json"):
    """Load reward history from a JSON file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            reward_history = json.load(f)
        print(f"Reward history loaded from {filename}")
        return reward_history
    else:
        print(f"No reward history file found. Starting fresh.")
        return []

# Update the main function
def main():
    env = gym.make("ALE/Frogger-v5", obs_type='grayscale', frameskip=(1, 4))
    num_actions = env.action_space.n
    input_shape = (1, 128, 128)

    model = A2CModel(input_shape, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Prompt user to choose whether to load a checkpoint or start fresh
    start_fresh = input("Do you want to start training from scratch? (y/n): ").lower() == 'y'
    
    if not start_fresh:
        if not load_checkpoint(model, optimizer, filename="a2c_checkpoint.pth"):
            print("No checkpoint found. Starting training from scratch.")
    
    # Load or initialize reward tracking
    reward_history = load_rewards()
    episode_rewards = deque(maxlen=100)
    replay_buffer = ReplayBuffer()

    episode = len(reward_history) * 5  # Adjust episode count based on reward history
    try:
        while True:  # Infinite loop
            state, _ = env.reset()
            state = preprocess_state(state)
            episode_reward = 0
            done = False
            action_count = 0
            count_per_action = [0] * num_actions

            while not done:
                action_probs, _ = model(state)
                action = torch.multinomial(action_probs, 1).item()
                count_per_action[action] += 1
                action_count += 1

                next_state, reward, done, _, _ = env.step(action)
                next_state = preprocess_state(next_state)
                replay_buffer.add(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

            print(f"Summary episode {episode}, main loop")
            print_action_dist(action_count, count_per_action, num_actions)
            episode_rewards.append(episode_reward)

            if episode % 5 == 0 and len(replay_buffer) >= 64:
                train_a2c(env, model, optimizer, replay_buffer)

            if episode % 50 == 0:
                save_checkpoint(model, optimizer, filename="a2c_checkpoint.pth")

            # Calculate and save the average reward every 5 episodes
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(list(episode_rewards)[-5:])
                reward_history.append(avg_reward)
                save_rewards(reward_history)
                print(f"Episode {episode}, Average Reward (last 5): {avg_reward:.2f}")

            episode += 1
    except KeyboardInterrupt:
        print("Training interrupted. Saving final checkpoint and reward history...")
        save_checkpoint(model, optimizer, filename="a2c_checkpoint.pth")
        save_rewards(reward_history)
        env.close()

if __name__ == "__main__":
    main()
