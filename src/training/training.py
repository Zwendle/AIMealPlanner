import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import pickle
import sys
import os
import matplotlib.pyplot as plt

ALPHA = 0.1       
GAMMA = 0.95   
EPSILON = 0.4
DECAY_RATE = 0.999995   
MAX_STEPS = 100
EPISODES = 750000
train_flag = 'train' in sys.argv
      
def make_state(ingredients, goal, df):
    total_cal = 0
    total_prot = 0
    total_carb = 0
    total_fat = 0
    total_price = 0
    
    for i in ingredients:
        row = df.loc[df['name_clean'] == i]
        if row.empty:
            print(f"Warning: Ingredient '{i}' not found in dataframe, skipping")
            continue
        
        total_cal += row['calories'].values[0]
        total_prot += row['protein'].values[0]
        total_carb += row['carbs'].values[0]
        total_fat += row['fat'].values[0]
        total_price += row['cost_per_serving'].values[0]

    def ratio_bucket(total, target):
        if target <= 0:
            return 0
        r = total / target
        if r < 0.6: return -2
        if r < 0.9: return -1
        if r < 1.1: return  0
        if r < 1.4: return  1
        return 2

    cal_b  = ratio_bucket(total_cal,  goal['target_calories'])
    prot_b = ratio_bucket(total_prot, goal['target_protein'])
    carb_b = ratio_bucket(total_carb, goal['target_carbs'])
    fat_b  = ratio_bucket(total_fat,  goal['target_fat'])
    price_b = ratio_bucket(total_price, goal['target_price'])

    size_b = len(ingredients) - 5

    return (
        cal_b,
        prot_b,
        carb_b,
        fat_b,
        price_b,
        size_b,
        int(goal['vegetarian_diet'])
    )

# used ChatGPT for this function
def parse_number(value):
   if isinstance(value, (int, float)):
       return value
   if isinstance(value, str):
       cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
       return float(cleaned) if cleaned else 0
   return 0


def sample_goal(df):
   all_ingredients = df['name_clean'].tolist()
  
   diet_type = random.choice(['high_protein', 'keto', 'low_carb', 'balanced'])
  
   if diet_type == 'high_protein':
       cal, prot, carbs, fat = 2000, 120, 200, 70
   elif diet_type == 'keto':
       cal, prot, carbs, fat = 2000, 100, 30, 120
   elif diet_type == 'low_carb':
       cal, prot, carbs, fat = 2000, 80, 100, 90
   else:
       cal, prot, carbs, fat = 2000, 50, 275, 78


   cal = int(cal * random.uniform(0.9, 1.1))
   prot = int(prot * random.uniform(0.9, 1.1))


   num_pantry = random.randint(5, 8)
   pantry = random.sample(all_ingredients, num_pantry)
  
   return {
       "target_calories": cal // 2,  # Per meal
       "target_protein": prot // 2,
       "target_carbs": carbs // 2,
       "target_fat": fat // 2,
       "vegetarian_diet": random.random() < 0.2,  # 20% vegetarian chance
       "target_price": random.randint(5, 10),  # Per meal
       "pantry": pantry
   }

## Keeps a max two ingredients per category constraint, and prioritizes same-category swaps
def get_actions(all_ingredients, current, df, ing_to_cat, max_per_category=2):
    category_counts = {}
    # count current categories
    for ing in current:
        category = ing_to_cat[ing] 
        category_counts[category] = category_counts.get(category, 0) + 1
    
    actions = []
    
    for out_ing in current:
        # get category of outgoing ingredient
        out_cat = ing_to_cat[out_ing]
        
        same_cat = []
        other_cat = []
        
        for in_ing in all_ingredients:
            if in_ing in current:
                continue
                
            # category of incoming ingredient
            in_cat = ing_to_cat[in_ing]
            
            
            # new_counts = category_counts.copy()
            # new_counts[out_cat] -= 1
            # new_counts[in_cat] = new_counts.get(in_cat, 0) + 1
            
            # if all(count <= max_per_category for count in new_counts.values()):
            if in_cat == out_cat: 
                same_cat.append((out_ing, in_ing))
            else:
                other_cat.append((out_ing, in_ing))
        
        actions.extend(same_cat)
        actions.extend(other_cat) # prioritize same category swaps
    
    return actions

def apply_action(ingredients, action):
   out_ing, in_ing = action
   new_list = ingredients.copy()
   new_list.remove(out_ing)
   new_list.append(in_ing)
   return new_list


def calculate_reward(df, ingredients, pantry_ingredients, target_calories, target_protein, vegetarian_diet, target_carbs, target_fat, target_price, servings_dict=None):
   """
   Calculate reward for a meal.
  
   Args:
       ingredients: List of ingredient names OR dict mapping ingredient -> servings_used
       servings_dict: Optional dict mapping ingredient -> servings_used (if ingredients is a list)
                     If ingredients is already a dict, this is ignored.
   """
   # Handle backward compatibility: if ingredients is a list, convert to dict
   if isinstance(ingredients, list):
       if servings_dict is None:
           # Default to 1 serving per ingredient
           servings_dict = {ing: 1.0 for ing in ingredients}
       ingredient_list = ingredients
   else:
       # ingredients is already a dict
       servings_dict = ingredients
       ingredient_list = list(servings_dict.keys())
  
   total_calories = 0
   total_protein = 0
   total_carbs = 0
   total_fats = 0
   total_price = 0
   in_pantry = 0
   vegetarian = True
  
   for ingredient in ingredient_list:
       row = df[df['name_clean'] == ingredient].iloc[0]
       servings_used = servings_dict.get(ingredient, 1.0)
      
       total_calories += parse_number(row.get('calories', 0)) * servings_used
       total_protein += parse_number(row.get('protein', 0)) * servings_used
       total_carbs += parse_number(row.get('carbs', 0)) * servings_used
       total_fats += parse_number(row.get('fat', 0)) * servings_used
       total_price += parse_number(row.get('cost_per_serving', 0)) * servings_used
      
       in_pantry += 1 if ingredient in pantry_ingredients else 0 
      
       is_meat = row.get('is_meat', False)
       if is_meat or row.get('category') == 'Meat & Seafood':
           vegetarian = False
  
   ingredient_score = in_pantry / len(ingredient_list) if ingredient_list else 0
   if vegetarian_diet and not vegetarian:
       ingredient_score *= 0.3
      
   def deviation_score(total, target):
       if target == 0:
           return 1
       return np.exp(-abs(total - target) / target)


   calorie_score = deviation_score(total_calories, target_calories)
   protein_score  = deviation_score(total_protein, target_protein)
   carb_score     = deviation_score(total_carbs, target_carbs)
   fat_score      = deviation_score(total_fats, target_fat)
  
   nutrition_score = np.mean([calorie_score, protein_score, carb_score, fat_score])
  
   if target_price == 0:
       cost_score = 1
   else:
       ratio = total_price / target_price
       cost_score = 1 if ratio <= 1 else np.exp(-(ratio - 1))
  
   # Weights: Utilization (45%), Nutrition (35%), Cost (20%)
   reward = 0.45 * ingredient_score + 0.35 * nutrition_score + 0.20 * cost_score
   return reward
 
def train(df):
    Q = {}   
    global EPSILON
    all_ingredients = df['name_clean'].tolist()
    rewards_per_ep = []
    ingredient_category_map = dict(zip(df['name_clean'], df['category']))

    for episode in tqdm(range(EPISODES)):
        total_reward = 0
        num_to_pick = random.randint(5, 8)
        ingredients = random.sample(all_ingredients, num_to_pick)
        goal = sample_goal(df)
        for step in range(MAX_STEPS):
            state = make_state(ingredients, goal, df)
            actions = get_actions(all_ingredients, ingredients, df, ingredient_category_map)
            if len(actions) == 0:
                print(f"Episode {episode}, Step {step}: No valid actions, ending episode early")
                break  # Exit the step loop, move to next episode

            if random.random() < EPSILON:
               action = random.choice(actions)
            else:
               q_vals = [Q.get((state, a), 0) for a in actions]
               action = actions[int(np.argmax(q_vals))]
            
            next_ingredients = apply_action(ingredients, action)
            next_state = make_state(next_ingredients, goal, df)


           # For training, use default 1 serving per ingredient
            servings_dict = {ing: 1.0 for ing in next_ingredients}
            reward = calculate_reward(
               df,
               next_ingredients,
               goal["pantry"],
               goal["target_calories"],
               goal["target_protein"],
               goal["vegetarian_diet"],
               goal["target_carbs"],
               goal["target_fat"],
               goal["target_price"],
               servings_dict=servings_dict
           )
            total_reward += reward

            old_q = Q.get((state, action), 0)
            next_actions = get_actions(all_ingredients, next_ingredients, df, ingredient_category_map)
            future_q = max([Q.get((next_state, a), 0) for a in next_actions]) if next_actions else 0
            Q[(state, action)] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)


            ingredients = next_ingredients
        rewards_per_ep.append(total_reward)
            
        EPSILON *= DECAY_RATE
          
          
    plt.figure()
    plt.plot(rewards_per_ep, color='steelblue', linewidth=1, label='Reward per Episode')
    plt.title(f"Training Performance", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    window = max(10, EPISODES // 100)
    running_avg = np.convolve(rewards_per_ep, np.ones(window) / window, mode='valid')
    plt.plot(range(window - 1, EPISODES), running_avg, color='darkorange', linewidth=2.5, label='Episode Running Average')
    plt.legend(fontsize=12)
    plt.show()

    plt.savefig(f"rewards.png", dpi=300)
 
    os.makedirs('data', exist_ok=True) # run out of /data instead
    with open('data/Q_table.pickle', 'wb') as handle:
       pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_meal(df, goal, Q_table, steps=20, num_to_pick=None, ingredients=None):
    if num_to_pick is None:
        num_to_pick = random.randint(5, 8)
        
    if ingredients is None:
        ingredients = {}
        
    ingredient_category_map = dict(zip(df['name_clean'], df['category']))

    all_ingredients = df['name_clean'].tolist()

    # Use num_servings when available, defaulting missing/NaN values to 1.0
    if 'num_servings' in df.columns:
        servings_series = df['num_servings'].fillna(1.0)
    else:
        # Fallback: everyone gets 1.0 serving
        servings_series = pd.Series(1.0, index=df.index)
    default_servings = dict(zip(df['name_clean'], servings_series))

    current_meal = {}

    if ingredients:
        input_names = list(ingredients.keys())
        if len(input_names) > num_to_pick:
            chosen_names = random.sample(input_names, num_to_pick)
            current_meal = {name: ingredients[name] for name in chosen_names}
        else:
            current_meal = ingredients.copy()

    current_names = list(current_meal.keys())
    slots_needed = num_to_pick - len(current_names)
    
    if slots_needed > 0:
        candidates = list(set(all_ingredients) - set(current_names))
        if candidates:
            actual_needed = min(slots_needed, len(candidates))
            new_items = random.sample(candidates, actual_needed)
            for item in new_items:
                current_meal[item] = 1.0

    working_names = list(current_meal.keys())

    for step in range(steps):
        state = make_state(working_names, goal, df)
        actions = get_actions(all_ingredients, working_names, df, ingredient_category_map)

        if not actions:
            break

        q_vals = [Q_table.get((state, a), 0) for a in actions]
        max_q = max(q_vals) if q_vals else 0
        best_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
        best_action = random.choice(best_actions)


        working_names = apply_action(working_names, best_action)

    final_meal = {}
    for name in working_names:
        if name in current_meal:
            final_meal[name] = current_meal[name]
        else:
            final_meal[name] = 1.0
    return final_meal      
           
def load_model(path='data/Q_table.pickle'):
   with open(path, 'rb') as handle:
       return pickle.load(handle)


def model_exists(path='data/Q_table.pickle'):
   return os.path.exists(path)

def analyze_qtable(Q_table, df):
    """Analyze Q-table coverage and sparsity"""
    
    # Basic stats
    total_entries = len(Q_table)
    unique_states = set(state for state, action in Q_table.keys())
    unique_actions = set(action for state, action in Q_table.keys())
    
    print(f"\n=== Q-Table Analysis ===")
    print(f"Total Q-table entries: {total_entries:,}")
    print(f"Unique states seen: {len(unique_states):,}")
    print(f"Unique actions seen: {len(unique_actions):,}")
    print(f"Avg actions per state: {total_entries / len(unique_states):.1f}")
    
    # Calculate theoretical state space
    # State = (cal_b, prot_b, carb_b, fat_b, price_b, size_b, vegetarian)
    # Each macro bucket: 5 values (-2, -1, 0, 1, 2)
    # size_b: varies based on ingredient count (typically -3 to 3 = 7 values)
    # vegetarian: 2 values (0, 1)
    theoretical_states = 5 * 5 * 5 * 5 * 5 * 7 * 2
    
    # Calculate theoretical action space
    n_ingredients = len(df)
    avg_meal_size = 6  # typically 5-8 ingredients
    actions_per_state = avg_meal_size * (n_ingredients - avg_meal_size)
    theoretical_actions = theoretical_states * actions_per_state
    
    print(f"\n=== State Space ===")
    print(f"Theoretical states: {theoretical_states:,}")
    print(f"States covered: {len(unique_states) / theoretical_states * 100:.2f}%")
    
    print(f"\n=== Action Space ===")
    print(f"Ingredients in dataset: {n_ingredients}")
    print(f"~Actions per state: {actions_per_state:,}")
    print(f"Theoretical (state,action) pairs: {theoretical_actions:,}")
    print(f"Coverage: {total_entries / theoretical_actions * 100:.4f}%")
    
    # Analyze Q-value distribution
    q_values = list(Q_table.values())
    print(f"\n=== Q-Value Distribution ===")
    print(f"Min Q: {min(q_values):.4f}")
    print(f"Max Q: {max(q_values):.4f}")
    print(f"Mean Q: {np.mean(q_values):.4f}")
    print(f"Std Q: {np.std(q_values):.4f}")
    
    # Plot distributions
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Q-value distribution
    axes[0, 0].hist(q_values, bins=50, edgecolor='black')
    axes[0, 0].set_title('Q-Value Distribution')
    axes[0, 0].set_xlabel('Q-Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Actions per state distribution
    state_action_counts = {}
    for state, action in Q_table.keys():
        state_action_counts[state] = state_action_counts.get(state, 0) + 1
    
    counts = list(state_action_counts.values())
    axes[0, 1].hist(counts, bins=50, edgecolor='black')
    axes[0, 1].set_title('Actions per State Distribution')
    axes[0, 1].set_xlabel('Number of Actions')
    axes[0, 1].set_ylabel('Number of States')
    
    # State component distributions
    state_components = {
        'cal': [], 'prot': [], 'carb': [], 'fat': [], 
        'price': [], 'size': [], 'veg': []
    }
    for state in unique_states:
        state_components['cal'].append(state[0])
        state_components['prot'].append(state[1])
        state_components['carb'].append(state[2])
        state_components['fat'].append(state[3])
        state_components['price'].append(state[4])
        state_components['size'].append(state[5])
        state_components['veg'].append(state[6])
    
    axes[1, 0].hist([state_components['cal'], state_components['prot'], 
                     state_components['carb'], state_components['fat']], 
                    bins=range(-3, 4), label=['Cal', 'Prot', 'Carb', 'Fat'],
                    alpha=0.7)
    axes[1, 0].set_title('Macro Nutrient Buckets Visited')
    axes[1, 0].set_xlabel('Bucket Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Coverage over theoretical space
    coverage_data = [
        len(unique_states) / theoretical_states * 100,
        100 - (len(unique_states) / theoretical_states * 100)
    ]
    axes[1, 1].pie(coverage_data, labels=['Covered', 'Uncovered'], 
                   autopct='%1.2f%%', startangle=90)
    axes[1, 1].set_title('State Space Coverage')
    
    plt.tight_layout()
    plt.savefig('data/qtable_analysis.png', dpi=150)
    print(f"\nSaved visualization to data/qtable_analysis.png")
    plt.close()

if __name__ == "__main__":
   df = pd.read_csv("data/ingredients_final.csv")
   required_cols = ["carbs", "fat", "protein", "calories", "cost_per_serving"]

   df = df.dropna(subset=required_cols)
   for col in required_cols:
       df = df[df[col].astype(str).str.strip() != ""]
      
   print("Rows after filtering:", len(df))
  
   if train_flag:
       train(df)
   else:
       if model_exists('data/Q_table.pickle'):
           Q_table = load_model('data/Q_table.pickle')
           analyze_qtable(Q_table, df)
       else:
           print("Model not found, training...")
           train(df)
           Q_table = load_model('data/Q_table.pickle')
       print(f"The length of Q table is: {len(Q_table)}")
       # change
       goal = sample_goal(df)
      
       print("GOAL")
       print(goal)
       final_ingredients_dict = generate_meal(df, goal, Q_table)
       print("Ingredients with servings:", final_ingredients_dict)
       score = calculate_reward(df, final_ingredients_dict, goal["pantry"], goal["target_calories"], goal["target_protein"], goal["vegetarian_diet"], goal["target_carbs"], goal["target_fat"], goal["target_price"])
       print(score)

