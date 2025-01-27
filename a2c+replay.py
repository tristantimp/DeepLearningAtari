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
matplotlib.use('TkAgg') # needed for visualisation apparently


def calculate_shaped_reward(env, current_position, previous_position, reached_goal):
    """
    Calculate shaped reward for Freeway based on progress, collisions, and goal.
    """
    # Progress reward: +1 for moving closer to the top, -0.1 for moving back
    progress_reward = 0.1 if current_position > previous_position else -0.1

    # Optional: Penalize staying idle (same position)
    if current_position == previous_position:
        progress_reward -= 0.05

    # Large reward for reaching the goal
    goal_reward = 1.0 if reached_goal else 0.0

    # Combine rewards
    total_reward = progress_reward + goal_reward
    return total_reward


def visualize_state(state):

    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()

    # Remove batch dimension if it exists
    if len(state.shape) == 4:  # (batch_size, channels, height, width)
        state = state[0]

    # Visualize the state
    plt.imshow(state[0], cmap='gray')  # Only visualize the first channel
    plt.title("State")
    plt.axis('off')
    plt.show()

def print_action_dist(action_count, count_per_action, num_actions):
    for action in range(num_actions):
        print(f"Model performed ACTION {action} {count_per_action[action] / action_count}%")
    # print("\n")

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

# Define the A2C model
class A2CModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A2CModel, self).__init__()

        # Input shape is now (1, height, width) for single frame
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Extra layer?
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the size of the flattened output after Conv layers (input size for actor and critic!)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Use the correct input shape here
            n_flatten = self.feature_extractor(dummy_input).shape[1]

        # Actor and Critic networks
        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)  # prob distribution
        )

        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)  # Extract features (main network)
        action_probs = self.actor(features)  # Actor computes probabilities
        value = self.critic(features)  # Critic judges state
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
def train_a2c(env, model, optimizer, replay_buffer, batch_size=64, num_steps=10, gamma=0.99):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    # Convert to tensors
    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)
    # Forward pass
    action_probs, values = model(states)
    next_action_probs, next_values = model(next_states)
    # Compute returns
    returns = rewards + gamma * next_values.squeeze() * (1 - dones)
    advantage = returns - values.squeeze()
    # print(f"Critic Values: {values.squeeze().detach().numpy()}")
    # print(f"Rewards: {rewards}")
    # print(f"Returns: {returns}")

    # Actor loss
    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
    actor_loss = -(log_probs * advantage.detach()).mean()
    # Critic loss
    critic_loss = F.mse_loss(values.squeeze(), returns.detach())
    # Total loss
    loss = actor_loss + critic_loss
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Loss: {loss.item():.4f}")


# Main training script
def main():
    # Initialize environment and model
    env = gym.make("ALE/Frogger-v5", obs_type='grayscale', frameskip=(1, 4))
    num_actions = env.action_space.n
    # changed this to 128x128 clearer features
    input_shape = (1, 128, 128)

    model = A2CModel(input_shape, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Track performance
    episode_rewards = deque(maxlen=100)
    replay_buffer = ReplayBuffer()

    # Train for a number of episodes
    for episode in range(1000):
        state, _ = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        done = False
        action_count = 0
        # for debug
        count_per_action = [0] * num_actions
        while not done:
            action_probs, _ = model(state)
            # Pick action weighted by probability (so it can also pick
            # an action with lower probability
            action = torch.multinomial(action_probs, 1).item()
            count_per_action[action] = count_per_action[action]+1
            action_count += 1
            # Step in the environment
            next_state, reward, done, _, _ = env.step(action)

            next_state = preprocess_state(next_state)
            # commented for now, nice for debugging to see each frame
            # visualize_state(next_state)  # Visualize the state
            replay_buffer.add(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
        # debugging
        print(f"Summary episode {episode}, main loop")
        print_action_dist(action_count, count_per_action, num_actions)
        episode_rewards.append(episode_reward)

        # Train the model every X episodes
        if episode % 5 == 0 and len(replay_buffer) >= 64:
            train_a2c(env, model, optimizer, replay_buffer)

        # Print average reward every 10 episodes
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
