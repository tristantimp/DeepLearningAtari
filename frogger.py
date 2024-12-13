import gymnasium as gym
import ale_py
import time

env = gym.make("ALE/Frogger", render_mode = "human")

state, _ = env.reset()

# Run a few steps to visualize the game
for _ in range(100):  # Run for 1000 steps
    env.render()  # Render the current game frame
    action = env.action_space.sample()  # Random action
    state, reward, done, truncated, info = env.step(action)  # Take the action
    
    if done:
        print("Game Over!")
        break
    
    time.sleep(0.02)  # Slow down to see the game progress

# Close the environment after the game is over
env.close()