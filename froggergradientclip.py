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
import json 
matplotlib.use('TkAgg')  # Needed for visualization 

# 1. Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    tensor_state = torch.tensor(resized_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Move to device
    if torch.isnan(tensor_state).any() or torch.isinf(tensor_state).any():
        print("Invalid state detected during preprocessing!")
        return torch.zeros_like(tensor_state)  # Replace invalid state with zeros
    return tensor_state

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
            dummy_input = torch.zeros(1, *input_shape).to(device)  # Move dummy input to device
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
        x = x.to(device)  # Ensure input is on the correct device
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        action_probs = torch.clamp(action_probs, min=1e-6, max=1 - 1e-6)  # Add numerical stability
        value = self.critic(features)
        return action_probs, value

def compute_returns(rewards, dones, gamma):
    returns = []
    G = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - done)
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

def train_a2c(model, optimizer, states, actions, rewards, dones, gamma=0.99):
    states = torch.cat(states).to(device)  # Move states to device
    actions = torch.tensor(actions, dtype=torch.long).to(device)  # Move actions to device
    dones = torch.tensor(dones, dtype=torch.float32).to(device)  # Move dones to device

    # Forward pass
    action_probs, values = model(states)

    # Compute returns and advantages
    returns = compute_returns(rewards, dones, gamma).to(device)
    advantage = returns - values.squeeze()

    # Actor loss
    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
    actor_loss = -(log_probs * advantage.detach()).mean()

    # Critic loss
    critic_loss = F.mse_loss(values.squeeze(), returns.detach())

    # Total loss
    loss = actor_loss + critic_loss

    # Backpropagation with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
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

def main():
    # Choose which environment to initialize with human rendering or without
    #env = gym.make("ALE/Frogger-v5", obs_type='grayscale', frameskip=(1, 4), render_mode="human")
    env = gym.make("ALE/Frogger-v5", obs_type='grayscale', frameskip=(1, 4), render_mode="rgb_array")

    num_actions = env.action_space.n
    input_shape = (1, 128, 128)
    model = A2CModel(input_shape, num_actions).to(device)  # Move model to device
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Load checkpoint from a2c_checkpoint_13500.pth
    checkpoint_file = "a2c_checkpoint_13500.pth"
    if not load_checkpoint(model, optimizer, filename=checkpoint_file):
        print(f"Checkpoint {checkpoint_file} not found! Starting training from scratch.")

    # Load reward history
    reward_history = load_rewards()
    episode_rewards = deque(maxlen=100)
    episode = len(reward_history) * 5

    try:
        while True:
            state, _ = env.reset()
            state = preprocess_state(state)
            done = False
            episode_reward = 0

            while not done:
                env.render()  # Render the environment in human mode
                action_probs, _ = model(state)
                action = torch.multinomial(action_probs, 1).item()
                next_state, reward, done, _, _ = env.step(action)
                state = preprocess_state(next_state)
                episode_reward += reward

            print(f"Episode {episode}, Reward: {episode_reward}")
            episode_rewards.append(episode_reward)
            episode += 1
    except KeyboardInterrupt:
        print("Training interrupted.")
        env.close()

if __name__ == "__main__":
    main()


