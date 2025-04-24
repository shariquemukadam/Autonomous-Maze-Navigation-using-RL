import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class DroneNavigationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(3,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6,), 
            dtype=np.float32
        )
        
        self.target = np.array([10, 10, 5], dtype=np.float32)
        self.state = None
        self.position_history = []  
        self.max_steps = 100
        self.current_step = 0
        self.prev_distance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = np.concatenate([
            np.zeros(3, dtype=np.float32),
            self.target.astype(np.float32)
        ]) / np.array([15, 15, 10, 15, 15, 10], dtype=np.float32)
        
        self.position_history = [self._denormalize_position(self.state[:3])]
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.target)
        
        return self.state.astype(np.float32), {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        
        denorm_pos = self._denormalize_position(self.state[:3])
        new_pos = denorm_pos + (action * np.array([0.5, 0.5, 0.3], dtype=np.float32))
        self.state[:3] = self._normalize_position(new_pos)
        self.position_history.append(new_pos.copy())
        self.current_step += 1

        current_pos = self._denormalize_position(self.state[:3])
        target_pos = self.target
        new_distance = np.linalg.norm(current_pos - target_pos)
        
        reward = float(self.prev_distance - new_distance)  # Convert to Python float
        self.prev_distance = new_distance
        
        if new_distance < 1.0:
            reward += 100.0
            
        reward -= 0.1
        
       
        terminated = bool(new_distance < 1.0)
        truncated = bool(self.current_step >= self.max_steps)
        
        return (
            self.state.astype(np.float32),  # Ensure float32
            reward,
            terminated,
            truncated,
            {}
        )

    def _normalize_position(self, pos):
        return pos.astype(np.float32) / np.array([15, 15, 10], dtype=np.float32)

    def _denormalize_position(self, norm_pos):
        return norm_pos.astype(np.float32) * np.array([15, 15, 10], dtype=np.float32)

    def render(self):
        plt.clf()
        positions = np.array(self.position_history)
        ax = plt.axes(projection='3d')
        ax.plot3D(positions[:,0], positions[:,1], positions[:,2], 'gray')
        ax.scatter3D(*self.target, color='red', s=100, label='Target')
        ax.scatter3D(*positions[-1], color='blue', s=100, label='Drone')
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 15)
        ax.set_zlim(0, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.01)

if __name__ == "__main__":
    
    env = DroneNavigationEnv()
    check_env(env)  
    
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )
    
    
    model.learn(total_timesteps=50000)
    
    
    obs, _ = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            break
    plt.show()
