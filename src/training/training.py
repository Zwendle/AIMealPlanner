import pandas as pd
import random
import json
import numpy as np
from tqdm import tqdm
import pickle
import sys
import os

ALPHA = 0.1        
GAMMA = 0.9        
EPSILON = 0.2      
MAX_STEPS = 200
EPISODES = 1000
Q = {}    
train_flag = 'train' in sys.argv
        

def make_state(ingredients):
    return tuple(sorted(ingredients))


def parse_number(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
        return float(cleaned) if cleaned else 0
    return 0

def sample_goal(df):
    all_ingredients = df['name'].tolist()
    
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


def get_actions(all_ingredients, current):
    actions = []
    for i in current:
        for j in all_ingredients:
            if j not in current:
                actions.append((i, j))
    return actions


def apply_action(ingredients, action):
    out_ing, in_ing = action
    new_list = ingredients.copy()
    new_list.remove(out_ing)
    new_list.append(in_ing)
    return new_list


def calculate_reward(df, ingredients, pantry_ingredients, target_calories, target_protein, vegetarian_diet, target_carbs, target_fat, target_price):
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fats = 0
    total_price = 0
    in_pantry = 0
    vegetarian = True
    
    for ingredient in ingredients:
        row = df[df['name'] == ingredient].iloc[0]
        
        total_calories += parse_number(row.get('calories', 0))
        total_protein += parse_number(row.get('protein', 0))
        total_carbs += parse_number(row.get('carbs', 0)) # i think this has to change to carbs_unit
        total_fats += parse_number(row.get('fat', 0)) # same here for fat_unit
        total_price += parse_number(row.get('price', 0)) # uh i think this should be cost per serving + retrain
        
        in_pantry += 1 if ingredient in pantry_ingredients else 0  
        
        is_meat = row.get('is_meat', False)
        if is_meat or row.get('category') == 'Meat & Seafood':
            vegetarian = False
    
    ingredient_score = in_pantry / len(ingredients)
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
    global Q
    all_ingredients = df['name'].tolist()

    for episode in tqdm(range(EPISODES)):

        num_to_pick = random.randint(5, 8)
        ingredients = random.sample(all_ingredients, num_to_pick)

        goal = sample_goal(df)

        for step in range(MAX_STEPS):

            state = make_state(ingredients)
            actions = get_actions(all_ingredients, ingredients)

            if random.random() < EPSILON:
                action = random.choice(actions)
            else:
                q_vals = [Q.get((state, a), 0) for a in actions]
                action = actions[int(np.argmax(q_vals))]

            next_ingredients = apply_action(ingredients, action)
            next_state = make_state(next_ingredients)

            reward = calculate_reward(
                df,
                next_ingredients,
                goal["pantry"],
                goal["target_calories"],
                goal["target_protein"],
                goal["vegetarian_diet"],
                goal["target_carbs"],
                goal["target_fat"],
                goal["target_price"]
            )

            old_q = Q.get((state, action), 0)
            future_q = max([Q.get((next_state, a), 0) for a in actions]) if actions else 0

            Q[(state, action)] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

            ingredients = next_ingredients
            
    os.makedirs('data', exist_ok=True) # run out of /data instead
    with open('data/Q_table.pickle', 'wb') as handle:
        pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_meal(df, goal, Q_table, steps=20):
    all_ingredients = df['name'].tolist()
    
    num_to_pick = random.randint(5, 8)
    ingredients = random.sample(all_ingredients, num_to_pick)

    for step in range(steps):
        state = make_state(ingredients)
        actions = get_actions(all_ingredients, ingredients)

        if not actions:
            break

        q_vals = [Q_table.get((state, a), 0) for a in actions]
        best_action = actions[int(np.argmax(q_vals))]

        ingredients = apply_action(ingredients, best_action)

    return ingredients

def load_model(path='data/Q_table.pickle'):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def model_exists(path='data/Q_table.pickle'):
    return os.path.exists(path)

if __name__ == "__main__":
    df = pd.read_csv("data/ingredients.csv")
    # No longer filtering by nutrition_json
    # df = df[df['nutrition_json'].notna() & df['nutrition_json'].astype(str).str.len().gt(2)]
    
    if train_flag:
        train(df)
    else:
        if model_exists('data/Q_table.pickle'):
            Q_table = load_model('data/Q_table.pickle')
        else:
            print("Model not found, training...")
            train(df)
            Q_table = Q
        
        # change
        goal = sample_goal(df)
        
        print("GOAL")
        print(goal)
        final_ingredients = generate_meal(df, goal, Q_table)
        print(final_ingredients)
        score = calculate_reward(df, final_ingredients, goal["pantry"], goal["target_calories"], goal["target_protein"], goal["vegetarian_diet"], goal["target_carbs"], goal["target_fat"], goal["target_price"])
        print(score)