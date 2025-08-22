"""
Data preprocessing for Dynamic Behavioral Cartography (DBC) experiment.
Handles behavior descriptor extraction from robot trajectories.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any
import brax
from brax import envs
from brax.envs import ant


def create_ant_environment():
    """Create the Brax ant environment for locomotion experiments."""
    env = envs.get_environment('ant')
    return env


def extract_behavior_descriptor(trajectory_data: Dict[str, Any], 
                              sampling_freq: int = 10,
                              duration: float = 2.0) -> np.ndarray:
    """
    Extract behavior descriptor from robot trajectory.
    
    Args:
        trajectory_data: Dictionary containing robot state trajectory
        sampling_freq: Sampling frequency in Hz (default: 10)
        duration: Duration in seconds to sample from end of episode (default: 2.0)
        
    Returns:
        Flattened behavior descriptor vector (240 dimensions)
        4 feet * 3 coordinates * 20 samples = 240 dimensions
    """
    foot_positions = trajectory_data.get('foot_positions', [])
    
    if len(foot_positions) == 0:
        return np.zeros(240)
    
    num_samples = int(sampling_freq * duration)
    total_timesteps = len(foot_positions)
    
    if total_timesteps < num_samples:
        padded_positions = np.zeros((num_samples, 4, 3))
        padded_positions[:total_timesteps] = foot_positions
        sampled_positions = padded_positions
    else:
        start_idx = total_timesteps - num_samples
        sampled_positions = foot_positions[start_idx:]
    
    behavior_descriptor = sampled_positions.flatten()
    
    return behavior_descriptor


def normalize_behavior_descriptors(descriptors: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize behavior descriptors to zero mean and unit variance.
    
    Args:
        descriptors: Array of behavior descriptors (N x 240)
        
    Returns:
        Normalized descriptors and normalization statistics
    """
    mean = np.mean(descriptors, axis=0)
    std = np.std(descriptors, axis=0)
    
    std = np.where(std == 0, 1.0, std)
    
    normalized = (descriptors - mean) / std
    
    stats = {
        'mean': mean,
        'std': std
    }
    
    return normalized, stats


def simulate_robot_episode(env, policy_params: np.ndarray, 
                          episode_length: int = 1000) -> Dict[str, Any]:
    """
    Simulate a single episode with the robot using given policy parameters.
    
    Args:
        env: Brax environment
        policy_params: Neural network parameters for the policy
        episode_length: Number of simulation steps
        
    Returns:
        Dictionary containing trajectory data and fitness
    """
    rng_key = jnp.array([0, 1], dtype=jnp.uint32)
    state = env.reset(rng=rng_key)
    
    trajectory = []
    foot_positions = []
    
    for step in range(episode_length):
        action = jnp.tanh(jnp.dot(state.obs, policy_params[:len(state.obs)]))
        
        state = env.step(state, action)
        
        trajectory.append({
            'obs': state.obs,
            'reward': state.reward,
            'done': state.done
        })
        
        foot_pos = extract_foot_positions_from_state(state)
        foot_positions.append(foot_pos)
    
    final_x_position = state.obs[0] if len(state.obs) > 0 else 0.0
    
    return {
        'trajectory': trajectory,
        'foot_positions': np.array(foot_positions),
        'fitness': final_x_position,
        'final_state': state
    }


def extract_foot_positions_from_state(state) -> np.ndarray:
    """
    Extract foot positions from Brax state.
    This is a simplified version - actual implementation depends on Brax state structure.
    
    Args:
        state: Brax environment state
        
    Returns:
        Array of foot positions (4 feet x 3 coordinates)
    """
    obs = state.obs
    
    torso_pos = obs[:3] if len(obs) >= 3 else np.array([0.0, 0.0, 0.0])
    
    foot_offsets = np.array([
        [0.5, 0.5, -0.5],   # Front right
        [0.5, -0.5, -0.5],  # Front left  
        [-0.5, 0.5, -0.5],  # Back right
        [-0.5, -0.5, -0.5]  # Back left
    ])
    
    foot_positions = torso_pos + foot_offsets
    
    return foot_positions


def create_initial_population(population_size: int = 100, 
                            genome_size: int = 256) -> np.ndarray:
    """
    Create initial population of random policy parameters.
    
    Args:
        population_size: Number of individuals in population
        genome_size: Size of each genome (policy parameter vector)
        
    Returns:
        Array of random policy parameters
    """
    return np.random.randn(population_size, genome_size) * 0.1


if __name__ == "__main__":
    print("Testing DBC preprocessing functions...")
    
    env = create_ant_environment()
    print(f"Created ant environment: {env}")
    
    dummy_trajectory = {
        'foot_positions': np.random.randn(50, 4, 3)
    }
    bd = extract_behavior_descriptor(dummy_trajectory)
    print(f"Behavior descriptor shape: {bd.shape}")
    
    descriptors = np.random.randn(10, 240)
    normalized, stats = normalize_behavior_descriptors(descriptors)
    print(f"Normalized descriptors shape: {normalized.shape}")
    
    print("Preprocessing tests completed successfully!")
