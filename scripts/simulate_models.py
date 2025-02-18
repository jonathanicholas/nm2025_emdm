import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random, math


@dataclass
class Episode:
    item_features: Tuple[str, str]
    reward: float

class EpisodicModel:
    def __init__(self, 
                 non_decision_time: float,
                 recall_time: float, 
                 recall_noise: float,
                 p_stop: float,
                 max_decision_time: float = 7.5):  
        self.non_decision_time = non_decision_time
        self.recall_time = recall_time
        self.recall_noise = recall_noise
        self.p_stop = p_stop
        self.max_decision_time = max_decision_time
        self.episodes: List[Episode] = []
        
    def encode(self, item_features: Tuple[str, str], reward: float):
        """Store an episode"""
        self.episodes.append(Episode(item_features, reward))
    
    def decide(self, offer_feature: Tuple[str, str]) -> Tuple[bool, float, int, float]:
        """Make a decision about an offer and return choice + RT + number of memories recalled + summed value"""
        elapsed_time = self.non_decision_time
        summed_value = 0
        n_recalled = 0
        
        # Randomly shuffle episodes to simulate random recall order
        recall_order = np.random.permutation(len(self.episodes))
        
        for idx in recall_order:
            # Check if we should stop (increasing probability with each memory)
            if np.random.random() < (self.p_stop * (n_recalled + 1)):
                break
                
            # Fixed recall time per memory
            next_recall_time = self.recall_time
            
            # Check if we've run out of time
            if elapsed_time + next_recall_time > self.max_decision_time:
                break
                
            episode = self.episodes[idx]
            elapsed_time += next_recall_time
            n_recalled += 1
            
            # Check if episode matches offer
            if offer_feature[0] in episode.item_features or offer_feature[1] in episode.item_features:
                # Add noise to recalled reward
                recalled_reward = episode.reward + np.random.normal(0, self.recall_noise)
                summed_value += recalled_reward
        
        return int(summed_value > 0), elapsed_time, n_recalled, summed_value  # Cast bool to int

class FeatureBasedModel:
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Inverse temperature parameter controlling choice randomness
                 Higher values = more deterministic choices
        """
        self.beta = beta
        self.feature_values: Dict[str, float] = {}
        
    def encode(self, item_features: Tuple[str, str], reward: float):
        """Update running sums for each feature"""
        for feature in item_features:
            if feature in self.feature_values:
                self.feature_values[feature] += reward
            else:
                self.feature_values[feature] = reward
    
    def decide(self, offer_feature: Tuple[str, str]) -> Tuple[bool, float]:
        """Make a decision about an offer using logistic choice rule"""
        # Check either feature (color or category)
        for feature in offer_feature:
            if feature in self.feature_values:
                # Logistic choice rule: P(accept) = 1 / (1 + exp(-beta * value))
                value = self.feature_values[feature]
                p_accept = 1 / (1 + np.exp(-self.beta * value))
                choice = int(np.random.random() < p_accept)
                return choice, 2
        
        return 0, 2


'''========================== Create Games =================================='''

def shuffle_array(array):
    """Shuffle array in place"""
    random.shuffle(array)

def get_items_for_games():
    """
    Get the item sets for games.
    Returns a list containing [full_item_set, full_dec_set]
    """
    # randomly determine which set to use
    curr_stims = random.randint(1, 5)
    
    if curr_stims == 1:
        full_item_set = [
            [
                "Food_Blue", "Food_Green", "Food_Red",
                "Object_Blue", "Object_Green", "Scene_Blue"
            ],
            [
                "Scene_Yellow", "Scene_Green", "Scene_Blue",
                "Food_Yellow", "Food_Green", "Animal_Yellow"
            ],
            [
                "Object_Red", "Object_Green", "Object_Yellow",
                "Animal_Red", "Animal_Green", "Scene_Red"
            ],
            [
                "Object_Blue", "Object_Yellow", "Object_Red",
                "Animal_Blue", "Animal_Yellow", "Food_Blue"
            ],
            [
                "Scene_Green", "Scene_Blue", "Scene_Yellow",
                "Animal_Green", "Animal_Blue", "Object_Green"
            ],
            [
                "Food_Red", "Food_Yellow", "Food_Blue",
                "Animal_Red", "Animal_Yellow", "Scene_Red"
            ],
            [
                "Scene_Yellow", "Scene_Green", "Scene_Red",
                "Food_Yellow", "Food_Green", "Object_Yellow"
            ],
            [
                "Animal_Red", "Animal_Blue", "Animal_Green",
                "Object_Red", "Object_Blue", "Food_Red"
            ]
        ]
        full_dec_set = [
            ["Object", "Green", "Red", "Scene", "Blue", "Food"],
            ["Scene", "Food", "Animal", "Yellow", "Blue", "Green"],
            ["Red", "Green", "Object", "Yellow", "Animal", "Scene"],
            ["Food", "Yellow", "Animal", "Red", "Blue", "Object"],
            ["Animal", "Blue", "Scene", "Green", "Yellow", "Object"],
            ["Yellow", "Animal", "Food", "Scene", "Blue", "Red"],
            ["Food", "Green", "Yellow", "Scene", "Object", "Red"],
            ["Red", "Object", "Green", "Blue", "Animal", "Food"]
        ]
    
    elif curr_stims == 2:
        full_item_set = [
            [
                "Object_Blue", "Object_Green", "Object_Yellow",
                "Animal_Blue", "Animal_Green", "Food_Blue"
            ],
            [
                "Scene_Red", "Scene_Blue", "Scene_Green",
                "Object_Red", "Object_Blue", "Food_Red"
            ],
            [
                "Food_Yellow", "Food_Red", "Food_Green",
                "Animal_Yellow", "Animal_Red", "Scene_Yellow"
            ],
            [
                "Animal_Yellow", "Animal_Blue", "Animal_Red",
                "Scene_Yellow", "Scene_Blue", "Object_Yellow"
            ],
            [
                "Food_Green", "Food_Yellow", "Food_Blue",
                "Animal_Green", "Animal_Yellow", "Scene_Green"
            ],
            [
                "Food_Green", "Food_Red", "Food_Yellow",
                "Scene_Green", "Scene_Red", "Object_Green"
            ],
            [
                "Scene_Red", "Scene_Yellow", "Scene_Blue",
                "Object_Red", "Object_Yellow", "Animal_Red"
            ],
            [
                "Object_Blue", "Object_Green", "Object_Red",
                "Animal_Blue", "Animal_Green", "Food_Blue"
            ]
        ]
        full_dec_set = [
            ["Object", "Blue", "Yellow", "Animal", "Food", "Green"],
            ["Food", "Red", "Green", "Blue", "Object", "Scene"],
            ["Scene", "Food", "Red", "Green", "Yellow", "Animal"],
            ["Object", "Animal", "Scene", "Yellow", "Red", "Blue"],
            ["Scene", "Animal", "Green", "Blue", "Food", "Yellow"],
            ["Green", "Scene", "Yellow", "Red", "Food", "Object"],
            ["Scene", "Yellow", "Red", "Object", "Blue", "Animal"],
            ["Food", "Object", "Blue", "Green", "Red", "Animal"]
        ]
    
    elif curr_stims == 3:
        full_item_set = [
            [
                "Animal_Yellow", "Animal_Blue", "Animal_Green",
                "Scene_Yellow", "Scene_Blue", "Food_Yellow"
            ],
            [
                "Food_Green", "Food_Blue", "Food_Red",
                "Object_Green", "Object_Blue", "Scene_Green"
            ],
            [
                "Scene_Red", "Scene_Yellow", "Scene_Blue",
                "Animal_Red", "Animal_Yellow", "Object_Red"
            ],
            [
                "Food_Red", "Food_Yellow", "Food_Green",
                "Object_Red", "Object_Yellow", "Animal_Red"
            ],
            [
                "Animal_Green", "Animal_Red", "Animal_Blue",
                "Scene_Green", "Scene_Red", "Object_Green"
            ],
            [
                "Food_Green", "Food_Red", "Food_Yellow",
                "Scene_Green", "Scene_Red", "Animal_Green"
            ],
            [
                "Object_Blue", "Object_Yellow", "Object_Green",
                "Scene_Blue", "Scene_Yellow", "Food_Blue"
            ],
            [
                "Object_Blue", "Object_Yellow", "Object_Red",
                "Animal_Blue", "Animal_Yellow", "Food_Blue"
            ]
        ]
        full_dec_set = [
            ["Animal", "Blue", "Yellow", "Scene", "Green", "Food"],
            ["Food", "Scene", "Blue", "Red", "Green", "Object"],
            ["Object", "Blue", "Yellow", "Animal", "Red", "Scene"],
            ["Animal", "Red", "Object", "Yellow", "Green", "Food"],
            ["Red", "Object", "Green", "Scene", "Blue", "Animal"],
            ["Green", "Food", "Red", "Yellow", "Scene", "Animal"],
            ["Scene", "Yellow", "Food", "Blue", "Green", "Object"],
            ["Blue", "Object", "Yellow", "Animal", "Food", "Red"]
        ]
    
    elif curr_stims == 4:
        full_item_set = [
            [
                "Object_Yellow", "Object_Red", "Object_Green",
                "Animal_Yellow", "Animal_Red", "Food_Yellow"
            ],
            [
                "Scene_Blue", "Scene_Yellow", "Scene_Green",
                "Food_Blue", "Food_Yellow", "Object_Blue"
            ],
            [
                "Food_Green", "Food_Red", "Food_Blue",
                "Scene_Green", "Scene_Red", "Animal_Green"
            ],
            [
                "Animal_Yellow", "Animal_Blue", "Animal_Red",
                "Scene_Yellow", "Scene_Blue", "Object_Yellow"
            ],
            [
                "Object_Green", "Object_Blue", "Object_Yellow",
                "Animal_Green", "Animal_Blue", "Food_Green"
            ],
            [
                "Scene_Red", "Scene_Blue", "Scene_Yellow",
                "Food_Red", "Food_Blue", "Object_Red"
            ],
            [
                "Object_Green", "Object_Blue", "Object_Red",
                "Animal_Green", "Animal_Blue", "Scene_Green"
            ],
            [
                "Food_Red", "Food_Yellow", "Food_Green",
                "Animal_Red", "Animal_Yellow", "Scene_Red"
            ]
        ]
        full_dec_set = [
            ["Animal", "Green", "Yellow", "Object", "Red", "Food"],
            ["Blue", "Yellow", "Green", "Object", "Food", "Scene"],
            ["Red", "Animal", "Scene", "Food", "Blue", "Green"],
            ["Object", "Red", "Yellow", "Scene", "Animal", "Blue"],
            ["Green", "Object", "Food", "Animal", "Blue", "Yellow"],
            ["Food", "Object", "Yellow", "Blue", "Red", "Scene"],
            ["Object", "Red", "Scene", "Green", "Blue", "Animal"],
            ["Green", "Animal", "Scene", "Food", "Yellow", "Red"]
        ]
    
    elif curr_stims == 5:
        full_item_set = [
            [
                "Scene_Blue", "Scene_Yellow", "Scene_Red",
                "Animal_Blue", "Animal_Yellow", "Food_Blue"
            ],
            [
                "Food_Yellow", "Food_Green", "Food_Red",
                "Scene_Yellow", "Scene_Green", "Object_Yellow"
            ],
            [
                "Animal_Yellow", "Animal_Blue", "Animal_Green",
                "Object_Yellow", "Object_Blue", "Food_Yellow"
            ],
            [
                "Object_Red", "Object_Green", "Object_Blue",
                "Animal_Red", "Animal_Green", "Scene_Red"
            ],
            [
                "Object_Green", "Object_Red", "Object_Yellow",
                "Scene_Green", "Scene_Red", "Animal_Green"
            ],
            [
                "Food_Blue", "Food_Red", "Food_Green",
                "Animal_Blue", "Animal_Red", "Scene_Blue"
            ],
            [
                "Scene_Green", "Scene_Blue", "Scene_Yellow",
                "Object_Green", "Object_Blue", "Food_Green"
            ],
            [
                "Food_Red", "Food_Yellow", "Food_Blue",
                "Animal_Red", "Animal_Yellow", "Object_Red"
            ]
        ]
        full_dec_set = [
            ["Scene", "Red", "Food", "Blue", "Animal", "Yellow"],
            ["Red", "Scene", "Food", "Yellow", "Object", "Green"],
            ["Blue", "Green", "Animal", "Yellow", "Food", "Object"],
            ["Animal", "Object", "Blue", "Scene", "Green", "Red"],
            ["Green", "Red", "Yellow", "Animal", "Object", "Scene"],
            ["Animal", "Red", "Food", "Scene", "Green", "Blue"],
            ["Food", "Blue", "Green", "Object", "Scene", "Yellow"],
            ["Animal", "Food", "Blue", "Object", "Red", "Yellow"]
        ]
    
    return [full_item_set, full_dec_set]

def generate_values_set(num_values):
    """Generate a sub-array of values meeting specified conditions"""
    while True:
        values_set = generate_random_sub_array(num_values)
        if is_valid_values_set(values_set, num_values):
            return values_set

def generate_random_sub_array(num_values):
    """Generate random sub-array of integers between -2 and 2 (excluding 0)"""
    sub_array = []
    for _ in range(num_values):
        while True:
            random_value = random.randint(-2, 2)
            if random_value != 0:
                sub_array.append(random_value)
                break
    return sub_array

def get_random_item(lst):
    """Get random item from list"""
    return random.choice(lst)

def is_valid_values_set(values_set, num_values):
    """Check if values sub-array meets specified conditions"""
    # Count positive and negative integers
    positive_count = sum(1 for value in values_set if value > 0)
    negative_count = sum(1 for value in values_set if value < 0)

    # Check if there are at least half positive and half negative integers
    if positive_count < num_values / 2 or negative_count < num_values / 2:
        return False

    if num_values == 6:
        # Check if at least one sum is less than 0
        condition1_met = any([
            sum(values_set[0:3]) < 0,
            sum(values_set[3:5]) < 0,
            values_set[5] < 0,
            sum([values_set[0], values_set[3], values_set[5]]) < 0,
            sum([values_set[1], values_set[4]]) < 0,
            values_set[2] < 0
        ])

        # Check if at least one sum is greater than 0
        condition2_met = any([
            sum(values_set[0:3]) > 0,
            sum(values_set[3:5]) > 0,
            values_set[5] > 0,
            sum([values_set[0], values_set[3], values_set[5]]) > 0,
            sum([values_set[1], values_set[4]]) > 0,
            values_set[2] > 0
        ])

        # Check that all sums are not equal to 0
        condition3_met = all([
            sum(values_set[0:3]) != 0,
            sum(values_set[3:5]) != 0,
            values_set[5] != 0,
            sum([values_set[0], values_set[3], values_set[5]]) != 0,
            sum([values_set[1], values_set[4]]) != 0,
            values_set[2] != 0
        ])

        return all([condition1_met, condition2_met, condition3_met])
    
    return False

def create_game(game_number, items_set, values_set, option_set, multiplier):
    """Create a game with pairs and values"""
    return {
        str(game_number): {
            "Pairs": items_set,
            "Values": values_set,
            "Options": option_set,
            "Multiplier": multiplier,
        }
    }

def initialize_games(n_games):
    # Create games and store them in a list
    games_array = []

    # Create the games
    full_item_set = get_items_for_games()

    # Add multiplier in this version, show it before or after
    multiplier_list = [
        ["Category"], ["Category"], ["Color"], ["Color"],
        ["Category"], ["Category"], ["Color"], ["Color"]
    ]
    shuffle_array(multiplier_list)
    
    multiplier_time = [
        ["Before", "After", "Before", "After", "Before", "After", "Before", "After"],
        ["After", "Before", "After", "Before", "After", "Before", "After", "Before"]
    ]
    shuffle_array(multiplier_time)
    multiplier_time = multiplier_time[0]

    for i in range(1, n_games + 1):
        multiplier_list[i-1].append(multiplier_time[i-1])
        curr_mult_list = multiplier_list[i-1]
        
        if curr_mult_list[0] == "Category":
            option_list = [item for item in full_item_set[1][i-1] 
                         if item not in ["Red", "Yellow", "Blue", "Green"]]
        else:
            option_list = [item for item in full_item_set[1][i-1]
                         if item not in ["Food", "Animal", "Object", "Scene"]]

        game = create_game(
            i,
            full_item_set[0][i-1],
            generate_values_set(6),
            option_list,
            curr_mult_list
        )
        games_array.append(game)

    # Convert the array to a dict with game names as keys
    games_object = {k: v for d in games_array for k, v in d.items()}
    
    return games_object

def simulate_trials(n_trials=1000):
    # Initialize models
    episodic = EpisodicModel(
        non_decision_time=1.5,      # Keep for first memory timing
        recall_time=0.5,            # Fixed time per memory
        recall_noise=0.5,             # Keep same reward noise level
        p_stop=0.1,                 # Probability of stopping after each memory
        max_decision_time=7.5       # Keep same max time
    )
    
    feature_based = FeatureBasedModel(beta=5.0)
    
    # Store results
    episodic_results = []
    feature_results = []
    
    for _ in range(n_trials):
        
        # Initialize a game exactly as in task.py
        games = initialize_games(8)
        
        # For each trial, randomly select one of the 8 games
        game_number = random.randint(1, 8)
        game = games[str(game_number)]
        
        # Extract the pairs, values, and decision options
        pairs = game["Pairs"]
        values = game["Values"]
        options = game["Options"]  # These are the features we should make decisions about
        
        # Initialize tracking of true values for features
        true_values = {}
        episodes = []
        
        # Process each pair and its value
        for i in range(len(pairs)):
            item = pairs[i]
            reward = values[i]
            
            # Split features
            feat_type, feat_color = item.split('_')
            
            # Add to episodes
            episodes.append(((feat_type, feat_color), reward))
            
            # Only track values for features in the decision set
            for feat in [feat_type, feat_color]:
                if feat in options:
                    if feat not in true_values:
                        true_values[feat] = 0
                    true_values[feat] += reward        
        # Reset models and encode episodes
        episodic.episodes = []
        feature_based.feature_values = {}
        
        for features, reward in episodes:
            episodic.encode(features, reward)
            feature_based.encode(features, reward)
        
        # Make decisions only for features in the options set
        for feature in options:
            # Episodic model
            choice, rt, n_recalled, recalled_value = episodic.decide((feature, ""))
            
            episodic_results.append({
                'model': 'episodic',
                'feature': feature,
                'choice': choice,
                'rt': rt,
                'n_memories': n_recalled,
                'true_value': true_values[feature],
                'recalled_value': recalled_value,
                'game_number': game_number
            })
            
            # Feature based model
            choice, rt = feature_based.decide((feature, ""))
            feature_results.append({
                'model': 'feature',
                'feature': feature,
                'choice': choice,
                'rt': rt,
                'true_value': true_values[feature],
                'recalled_value': recalled_value,
                'game_number': game_number
            })
    
    return pd.DataFrame(episodic_results), pd.DataFrame(feature_results)

# Run simulation
episodic_df, feature_df = simulate_trials(1000)

episodic_df.to_csv('data/episodic_model_sim.csv', index=False)
feature_df.to_csv('data/feature_model_sim.csv', index=False)