import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py
import os
import psutil
import time
import threading
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

env = gym.make("ALE/Freeway-v5" , obs_type="grayscale", render_mode = 'rgb_array', frameskip = (1,4))

# Define the actor and critic networks
actor = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='softmax')
])

critic = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Define optimizer and loss functions
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def monitor_memory():
    process = psutil.Process()  # Current process
    while True:
        memory_info = process.memory_info()
        print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")  # Print in MB
        time.sleep(1)

monitor_thread = threading.Thread(target=monitor_memory)
monitor_thread.daemon = True
monitor_thread.start()

# Main training loop
num_episodes = 10
gamma = 0.99
actor_losses = []
critic_losses = []

for episode in range(num_episodes):
    state, _ = env.reset()
    #print("State shape:", state.shape)
    episode_reward = 0

    with tf.GradientTape(persistent=True) as tape:
        for t in range(1, 1000):  # Limit the number of time steps

            # Choose an action using the actor
            flattened_state = state.flatten()
            #print("State shape", flattened_state.shape)
            action_probs = actor(np.array([flattened_state]))
            epsilon = 1e-4
            action_probs = tf.clip_by_value(action_probs, clip_value_min=epsilon, clip_value_max=1.0-epsilon)
            #print("action probability:", action_probs)
            action = np.random.choice(env.action_space.n, p=action_probs.numpy()[0])
            #print("Action:", action)

            # Take the chosen action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)

            # Compute the advantage
            state_value = critic(np.array([flattened_state]))[0, 0]
            #print("State value:", state_value)
            flattened_next_state = next_state.flatten()
            next_state_value = critic(np.array([flattened_next_state]))[0, 0]
            #print("Next state value:", next_state_value)
            advantage = reward + gamma * next_state_value - state_value

            # Compute actor and critic losses
            actor_loss = -tf.math.log(action_probs[0, action]) * advantage
            critic_loss = tf.square(advantage)
            #print("Actor loss:", actor_loss)
            #print("Critic loss:", critic_loss)
            actor_losses.append(actor_loss.numpy())
            critic_losses.append(critic_loss.numpy())
            episode_reward += reward

            # Update actor and critic
            actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
            critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        tape.reset()


        if done:
            break

    if episode % 1 == 0:
        print(f'Episode {episode}, Reward: {episode_reward}')

env.close()
print(actor.summary())
print(critic.summary())

plt.subplot(2, 1, 1)
plt.plot(actor_losses, label="Actor Loss", color="blue")
plt.title("Actor Loss over Episodes")
plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.legend()

# Plot critic loss
plt.subplot(2, 1, 2)
plt.plot(critic_losses, label="Critic Loss", color="red")
plt.title("Critic Loss over Episodes")
plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("actor_critic_loss.png") 
