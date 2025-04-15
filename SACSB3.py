import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# Network architecture
POLICY_HIDDEN_SIZES = [256, 256, 256]
VALUE_HIDDEN_SIZES = [256, 256, 256]

# Model save path
SAVE_PATH = "tmp"
os.makedirs(SAVE_PATH, exist_ok=True)

# Create training and testing environments
env_train = gym.make("LunarLander-v3",continuous=True)
env_test = gym.make("LunarLander-v3", render_mode="human", continuous=True)

policy_kwargs = dict(
    net_arch=dict(
        pi=POLICY_HIDDEN_SIZES,
        qf=VALUE_HIDDEN_SIZES
    )
)

# Improved callback to log rewards and losses from the logger
class LoggerDataCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = []
        # Loss metrics
        self.critic_losses = []
        self.actor_losses = []
        self.entropy_losses = []
        self.entropy_coefs = []
        # Update count for x-axis of loss plots
        self.update_counts = []
        
        # Previous values for change detection
        self.prev_episodes = 0
        self.timestep_count = []
        
    def _on_step(self) -> bool:
        # Check if the logger has updated values
        if hasattr(self.model, "logger") and self.model.logger is not None:
            logger_data = {}
            
            # Extract all values from the logger
            for key, value in self.model.logger.name_to_value.items():
                try:
                    logger_data[key] = float(value)
                except (ValueError, TypeError):
                    continue
            
            # Get current episode count
            current_episodes = logger_data.get("time/episodes", 0)
            
            # Check if a new episode was completed
            if current_episodes > self.prev_episodes:
                # Record episode reward and length
                if "rollout/ep_rew_mean" in logger_data:
                    self.episode_rewards.append(logger_data["rollout/ep_rew_mean"])
                
                if "rollout/ep_len_mean" in logger_data:
                    self.episode_lengths.append(logger_data["rollout/ep_len_mean"])
                
                # Record the episode number
                self.episode_count.append(current_episodes)
                
                # Record timestep at this episode completion
                self.timestep_count.append(logger_data.get("time/total_timesteps", self.num_timesteps))
                
                # Update previous episode count
                self.prev_episodes = current_episodes
                
                print(f"Episode {current_episodes} completed with mean reward: {logger_data.get('rollout/ep_rew_mean', 'N/A')}")
            
            # Record training losses
            if "train/critic_loss" in logger_data:
                self.critic_losses.append(logger_data["train/critic_loss"])
                self.update_counts.append(logger_data.get("train/n_updates", len(self.critic_losses)))
            
            if "train/actor_loss" in logger_data:
                self.actor_losses.append(logger_data["train/actor_loss"])
            
            if "train/ent_coef_loss" in logger_data:
                self.entropy_losses.append(logger_data["train/ent_coef_loss"])
            
            if "train/ent_coef" in logger_data:
                self.entropy_coefs.append(logger_data["train/ent_coef"])
        
        return True

# Function for safe smoothing that handles edge cases
def smooth_data(data, window_size=5):
    """Safely smooth data with a moving average window."""
    if len(data) < window_size:
        return data, range(len(data))  # Not enough data to smooth
    
    try:
        data_array = np.array(data, dtype=float)
        smoothed = np.convolve(data_array, np.ones(window_size)/window_size, mode='valid')
        # Return both the smoothed data and the corresponding x indices
        return smoothed, range(window_size-1, len(data))
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not smooth data due to error: {e}")
        return data, range(len(data))  # Return original data if conversion fails

# Create the model
model = SAC(
    "MlpPolicy",
    env_train,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./sac_tensorboard2/",
    learning_rate=0.0003,
    buffer_size=int(1e6),
    learning_starts=100,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto"
)
# model = SAC.load("C:/Project/tmp/sac_llv3.zip", env_train)
print("loaded weights")
# Create the callback instance
callback = LoggerDataCallback()

# Train the model
print("Starting training...")
model.learn(total_timesteps=1e5, log_interval=5, callback=callback)

print(f"Training complete! Episodes completed: {callback.prev_episodes}")

# Optional: Pause before saving
input("\n🎉 Training complete! Press Enter to save the model...")

# Save model
model.save(os.path.join(SAVE_PATH, "sac_llv3"))
print(f"✅ Model saved to {os.path.join(SAVE_PATH, 'sac_llv3.zip')}")

# Print summary of collected data
print(f"Collected data summary:")
print(f"- Episodes completed: {len(callback.episode_rewards)}")
print(f"- Loss updates recorded: {len(callback.critic_losses)}")

# Create plots of training data
# Figure 1: Episode rewards
plt.figure(figsize=(12, 6))
if callback.episode_rewards:
    # Plot raw episode rewards
    plt.plot(callback.episode_count, callback.episode_rewards, 'b-', alpha=0.3, label="Episode Rewards")
    
    # Plot smoothed rewards if enough data
    if len(callback.episode_rewards) > 5:
        window_size = min(5, max(2, len(callback.episode_rewards) // 10))
        smoothed_rewards, smooth_x = smooth_data(callback.episode_rewards, window_size)
        smooth_x_episodes = [callback.episode_count[i] for i in smooth_x]
        plt.plot(smooth_x_episodes, smoothed_rewards, 'r-', linewidth=2,
                label=f"Smoothed Rewards (window={window_size})")
    
    plt.xlabel("Episodes")
    plt.ylabel("Mean Episode Reward")
    plt.title("Mean Episode Reward During Training")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "episode_rewards.png"))
else:
    plt.text(0.5, 0.5, "No episode reward data collected", 
             horizontalalignment='center', verticalalignment='center')
    plt.savefig(os.path.join(SAVE_PATH, "episode_rewards.png"))

# Figure 2: Episode length
plt.figure(figsize=(12, 6))
if callback.episode_lengths:
    # Plot raw episode lengths
    plt.plot(callback.episode_count, callback.episode_lengths, 'g-', alpha=0.3, label="Episode Lengths")
    
    # Plot smoothed lengths if enough data
    if len(callback.episode_lengths) > 5:
        window_size = min(5, max(2, len(callback.episode_lengths) // 10))
        smoothed_lengths, smooth_x = smooth_data(callback.episode_lengths, window_size)
        smooth_x_episodes = [callback.episode_count[i] for i in smooth_x]
        plt.plot(smooth_x_episodes, smoothed_lengths, 'c-', linewidth=2,
                label=f"Smoothed Lengths (window={window_size})")
    
    plt.xlabel("Episodes")
    plt.ylabel("Mean Episode Length")
    plt.title("Mean Episode Length During Training")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "episode_lengths.png"))

# Figure 3: Critic and Actor Losses
plt.figure(figsize=(15, 10))

# Plot critic loss
plt.subplot(2, 1, 1)
if callback.critic_losses and callback.update_counts:
    # Ensure we have the same number of x and y values
    x_values = callback.update_counts[:len(callback.critic_losses)]
    plt.plot(x_values, callback.critic_losses, 'g-', alpha=0.3, label="Critic Loss")
    
    # Plot smoothed critic loss if enough data
    if len(callback.critic_losses) > 20:
        window_size = min(20, len(callback.critic_losses) // 10)
        smoothed_critic, smooth_x = smooth_data(callback.critic_losses, window_size)
        # Map smoothed indices back to update counts
        smooth_x_updates = [x_values[i] for i in smooth_x if i < len(x_values)]
        smoothed_critic = smoothed_critic[:len(smooth_x_updates)]  # Ensure sizes match
        if smooth_x_updates:
            plt.plot(smooth_x_updates, smoothed_critic, 'g-', linewidth=2,
                    label=f"Smoothed Critic Loss (window={window_size})")
    
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.title("Critic Loss During Training")
    plt.grid(True)
    plt.legend()
else:
    plt.text(0.5, 0.5, "No critic loss data collected", 
             horizontalalignment='center', verticalalignment='center')

# Plot actor loss and entropy
plt.subplot(2, 1, 2)
if callback.actor_losses and callback.update_counts:
    # Ensure we have the same number of x and y values
    x_values = callback.update_counts[:len(callback.actor_losses)]
    plt.plot(x_values, callback.actor_losses, 'r-', alpha=0.3, label="Actor Loss")
    
    # Plot smoothed actor loss if enough data
    if len(callback.actor_losses) > 20:
        window_size = min(20, len(callback.actor_losses) // 10)
        smoothed_actor, smooth_x = smooth_data(callback.actor_losses, window_size)
        # Map smoothed indices back to update counts
        smooth_x_updates = [x_values[i] for i in smooth_x if i < len(x_values)]
        smoothed_actor = smoothed_actor[:len(smooth_x_updates)]  # Ensure sizes match
        if smooth_x_updates:
            plt.plot(smooth_x_updates, smoothed_actor, 'r-', linewidth=2,
                    label=f"Smoothed Actor Loss (window={window_size})")
    
    plt.xlabel("Updates")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss During Training")
    plt.grid(True)
    plt.legend(loc="upper left")
    
    # Add entropy coefficient on secondary y-axis if available
    if callback.entropy_coefs:
        ax2 = plt.twinx()
        x_values_ent = callback.update_counts[:len(callback.entropy_coefs)]
        ax2.plot(x_values_ent, callback.entropy_coefs, 'b-', alpha=0.3, label="Entropy Coefficient")
        
        # Plot smoothed entropy coef if enough data
        if len(callback.entropy_coefs) > 20:
            window_size = min(20, len(callback.entropy_coefs) // 10)
            smoothed_ent, smooth_x = smooth_data(callback.entropy_coefs, window_size)
            # Map smoothed indices back to update counts
            smooth_x_updates = [x_values_ent[i] for i in smooth_x if i < len(x_values_ent)]
            smoothed_ent = smoothed_ent[:len(smooth_x_updates)]  # Ensure sizes match
            if smooth_x_updates:
                ax2.plot(smooth_x_updates, smoothed_ent, 'b-', linewidth=2,
                        label=f"Smoothed Entropy Coef (window={window_size})")
        
        ax2.set_ylabel("Entropy Coefficient")
        ax2.legend(loc="upper right")
else:
    plt.text(0.5, 0.5, "No actor loss data collected", 
             horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "training_losses.png"))

# Figure 4: Learning curve (reward vs timesteps)
plt.figure(figsize=(12, 6))
if callback.episode_rewards and callback.timestep_count:
    plt.plot(callback.timestep_count, callback.episode_rewards, 'b-', alpha=0.3, label="Rewards")
    
    # Plot smoothed rewards if enough data
    if len(callback.episode_rewards) > 5:
        window_size = min(5, max(2, len(callback.episode_rewards) // 10))
        smoothed_rewards, smooth_x = smooth_data(callback.episode_rewards, window_size)
        smooth_x_timesteps = [callback.timestep_count[i] for i in smooth_x if i < len(callback.timestep_count)]
        smoothed_rewards = smoothed_rewards[:len(smooth_x_timesteps)]  # Ensure sizes match
        if smooth_x_timesteps:
            plt.plot(smooth_x_timesteps, smoothed_rewards, 'r-', linewidth=2,
                    label=f"Smoothed Rewards (window={window_size})")
    
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.title("Learning Curve: Reward vs Timesteps")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "learning_curve.png"))

# Show all plots
plt.show()

# Run trained model in test environment
print("Running trained model in test environment...")
obs, info = env_test.reset()
try:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        if terminated or truncated:
            obs, info = env_test.reset()
except KeyboardInterrupt:
    print("Testing ended by user")
finally:
    env_test.close()