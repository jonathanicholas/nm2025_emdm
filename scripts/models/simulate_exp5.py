import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random, math
import itertools

@dataclass
class Episode:
    item_features: Tuple[str, str]
    reward: float

class EpisodicModel:
    def __init__(self, 
                 recall_time: float, 
                 recall_noise: float,
                 p_stop: float,
                 max_decision_time: float = 7.5,
                 temperature: float = 1.0):
        self.recall_time = recall_time
        self.recall_noise = recall_noise
        self.p_stop = p_stop
        self.max_decision_time = max_decision_time
        self.temperature = temperature
        self.episodes: List[Episode] = []
        
    def encode(self, item_features: Tuple[str, str], reward: float):
        """Store an episode"""
        self.episodes.append(Episode(item_features, reward))

    def decide(self, decision_features: List[int], all_memories: List[Tuple]) -> Tuple[bool, float, int, int, float]:
        elapsed_time = 0  # Start at 0 instead of non_decision_time
        summed_value = 0
        sampled_indices = []
        n_relevant_recalled = 0
        
        if len(self.episodes) == 0:
            return False, elapsed_time, 0, 0, 0
        
        # Calculate sampling probabilities using softmax
        logits = []
        for episode in self.episodes:
            memory_features = episode.item_features
            is_relevant = any(memory_features[feat_idx] == 1 for feat_idx in decision_features)
            logit = self.temperature if is_relevant else 0.0
            logits.append(logit)
        
        logits = np.array(logits)
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        # Track remaining memories available for sampling
        remaining_indices = list(range(len(self.episodes)))
        
        # Sample memories until stopping (without replacement)
        while remaining_indices and elapsed_time < self.max_decision_time:
            # Geometric stopping check
            if random.random() < self.p_stop:
                break
            
            # Update probabilities for remaining memories only
            current_probs = probabilities[remaining_indices]
            current_probs = current_probs / current_probs.sum()  # Renormalize
            
            # Sample a memory from remaining memories
            chosen_idx_in_remaining = np.random.choice(len(remaining_indices), p=current_probs)
            memory_idx = remaining_indices[chosen_idx_in_remaining]
            
            # Remove from remaining memories
            remaining_indices.remove(memory_idx)
            sampled_indices.append(memory_idx)
            
            episode = self.episodes[memory_idx]
            memory_features = episode.item_features
            
            # Check if this memory is relevant
            is_relevant = any(memory_features[feat_idx] == 1 for feat_idx in decision_features)
            if is_relevant:
                n_relevant_recalled += 1
            
            # Add reward
            noise = random.gauss(0, self.recall_noise)
            summed_value += (episode.reward + noise)
            
            # Add recall time
            elapsed_time += self.recall_time
        
        # Make random decision if no relevant memories were sampled
        if n_relevant_recalled == 0:
            choice = random.random() < 0.5
            summed_value = 0
        else:
            choice = summed_value > 0
        
        return choice, elapsed_time, len(sampled_indices), n_relevant_recalled, summed_value

class FeatureBasedModel:
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Inverse temperature parameter controlling choice randomness
                 Higher values = more deterministic choices
        """
        self.beta = beta
        self.feature_values: Dict[str, float] = {}
        
    def encode(self, memory_features: Tuple, reward: float):
        """Update running sums for each active feature"""
        for feat_idx, is_active in enumerate(memory_features):
            if is_active == 1:  # Only update if feature is active
                if feat_idx in self.feature_values:
                    self.feature_values[feat_idx] += reward
                else:
                    self.feature_values[feat_idx] = reward
    
    def decide(self, decision_features: List[int]) -> Tuple[bool, float]:
        # Sum up values for all relevant features
        total_value = 0
        for feat_idx in decision_features:
            if feat_idx in self.feature_values:
                total_value += self.feature_values[feat_idx]
        
        # Logistic choice rule
        p_accept = 1 / (1 + np.exp(-self.beta * total_value))
        choice = int(np.random.random() < p_accept)
        return choice, 2
    
def generate_memories(n_memories: int, n_features: int) -> Tuple[List[Tuple], List[int]]:
    """
    Generate memories with binary feature representations.
    
    Args:
        n_memories: Total number of memories to generate
        n_features: Number of binary features each memory has
    
    Returns:
        memories: List of tuples representing binary feature vectors
        rewards: List of reward values for each memory
    """
    memories = []
    rewards = []
    
    for _ in range(n_memories):
        # Generate binary feature vector (each feature is 0 or 1)
        features = tuple(np.random.choice([0, 1], size=n_features))
        memories.append(features)
        
        # Generate reward
        reward = random.choice([-2, -1, 1, 2])
        rewards.append(reward)
    
    return memories, rewards

def apply_memory_capacity(memories: List[Tuple], rewards: List[int], 
                         memory_capacity: float) -> Tuple[List[Tuple], List[int], int]:
    """
    Apply memory capacity constraint by randomly selecting a subset of memories.
    
    Args:
        memories: List of all generated memories
        rewards: List of all generated rewards  
        memory_capacity: Proportion of memories to retain (0.0 to 1.0)
    
    Returns:
        selected_memories: Subset of memories the agent actually remembers
        selected_rewards: Corresponding rewards
        n_actual_memories: Actual number of memories retained
    """
    total_memories = len(memories)
    n_actual_memories = int(total_memories * memory_capacity)
    n_actual_memories = max(1, min(n_actual_memories, total_memories))  # Ensure at least 1 memory
    
    # Randomly select which memories to retain
    selected_indices = np.random.choice(total_memories, size=n_actual_memories, replace=False)
    
    selected_memories = [memories[i] for i in selected_indices]
    selected_rewards = [rewards[i] for i in selected_indices]
    
    return selected_memories, selected_rewards, n_actual_memories

def generate_variable_relevance_decisions(memories: List[Tuple], n_decisions: int, 
                                        n_features: int) -> List[int]:
    """
    Generate decisions with varying relevance levels within the same trial.
    This creates a range of easy (high relevance) to hard (low relevance) decisions.
    """
    decisions = []
    
    # Create decisions with different relevance levels to avoid ceiling effects
    target_relevances = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    
    for i in range(n_decisions):
        # Cycle through different target relevance levels
        target_rel = target_relevances[i % len(target_relevances)]
        
        # Find feature that gives closest to target relevance
        best_feature = None
        best_diff = float('inf')
        
        for feat_idx in range(n_features):
            # Calculate actual relevance for this feature
            n_relevant = sum(1 for mem in memories if mem[feat_idx] == 1)
            actual_rel = n_relevant / len(memories) if len(memories) > 0 else 0
            
            # Find feature closest to target
            diff = abs(actual_rel - target_rel)
            if diff < best_diff:
                best_diff = diff
                best_feature = feat_idx
        
        decisions.append(best_feature)
    
    return decisions

def count_relevant_memories_by_feature(memories: List[Tuple], decision_feature: int) -> int:
    """Count how many memories have the asked-about feature."""
    return sum(1 for memory in memories if memory[decision_feature] == 1)

def simulate_experiment(n_trials_per_combo=1000, n_features=20, n_decisions=8):
    """
    Simulates experiment 5, storing individual trial data.
    
    Args:
        n_trials_per_combo: Number of trials per parameter combination
        n_features: Number of binary features per memory
        n_decisions: Number of decisions per trial
    """
    
    # Define the experimental regime
    n_memories = 21
    
    # Parameter grids
    p_stop_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    recall_time_values = [1]
    temperature_values = [0, 1, 2.5, 5, 10]
        
    all_results = []
    total_combinations = (len(p_stop_values) * len(recall_time_values) * 
                         len(temperature_values))

    print(f"Running detailed simulations with {total_combinations} parameter combinations...")
    print(f"Each combination will have {n_trials_per_combo} trials")
    print(f"Total trials: {total_combinations * n_trials_per_combo:,}")
    
    combination_count = 0
    
    for p_stop in p_stop_values:
        for recall_time in recall_time_values:
            for temperature in temperature_values:
                
                combination_count += 1
                
                # Initialize model for this parameter combination
                episodic = EpisodicModel(
                    recall_time=recall_time,
                    recall_noise=0.001,
                    p_stop=p_stop,
                    max_decision_time=7.5,
                    temperature=temperature
                )
                
                # Run trials for this parameter combination
                for trial in range(n_trials_per_combo):
                    # Generate memories and rewards for this trial
                    memories, rewards = generate_memories(n_memories, n_features)
                    
                    # Generate decision features
                    decisions = generate_variable_relevance_decisions(
                        memories, n_decisions, n_features
                    )
                    
                    # Reset model and encode episodes
                    episodic.episodes = []
                    
                    # Encode all memories
                    for memory, reward in zip(memories, rewards):
                        episodic.encode(memory, reward)
                    
                    # Calculate true values for each decision
                    true_values = {}
                    
                    for dec_idx, decision_feature in enumerate(decisions):
                        # True value based on all memories with the relevant feature
                        true_value = 0
                        for memory, reward in zip(memories, rewards):
                            if memory[decision_feature] == 1:  # Memory has the asked-about feature
                                true_value += reward
                        true_values[dec_idx] = true_value
                    
                    # Make decisions
                    for dec_idx, decision_feature in enumerate(decisions):
                        # Count relevant memories
                        n_available_relevant = count_relevant_memories_by_feature(memories, decision_feature)
                        
                        # Make decision
                        choice, rt, n_recalled, n_relevant_recalled, recalled_value = episodic.decide(
                            [decision_feature], memories
                        )
                        
                        # Store detailed results
                        all_results.append({
                            'n_memories': n_memories,
                            'p_stop': p_stop,
                            'recall_time': recall_time,
                            'temperature': temperature,
                            'trial': trial,
                            'decision': dec_idx,
                            'choice': choice,
                            'rt': rt,
                            'n_memories_recalled': n_recalled,
                            'n_relevant_recalled': n_relevant_recalled,
                            'n_available_relevant': n_available_relevant,
                            'true_value': true_values[dec_idx],
                            'recalled_value': recalled_value,
                            'n_features': n_features,
                            'n_decisions': n_decisions
                        })
                
                # Print progress update
                if combination_count % 5 == 0:
                    print(f"  Completed {combination_count}/{total_combinations} combinations...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    output_file = '../data/experiment_5_sim.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed simulations complete! Results saved to {output_file}")
    print(f"Final dataset shape: {results_df.shape}")
    
    return results_df

# Run the experiment 5 simulations
detailed_results = simulate_experiment(n_trials_per_combo=5000)