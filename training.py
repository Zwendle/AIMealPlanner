import pandas as pd
import random
import json
import numpy as np
from tqdm import tqdm
import pickle
import sys


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
    num_to_pick = random.randint(0, 25)
    ingredients = random.sample(all_ingredients, num_to_pick)
    
    return {
        "target_calories": random.randint(400, 900),
        "target_protein": random.randint(20, 80),
        "target_carbs": random.randint(20, 120),
        "target_fat": random.randint(10, 60),
        "vegetarian_diet": random.choice([True, False]),
        "target_price": random.randint(10, 30),
        "pantry": ingredients 
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
        value = df.loc[df['name'] == ingredient, 'nutrition_json'].iloc[0]
        if isinstance(value, str):
            nutrition = json.loads(value)
        else:
            nutrition = value    
        
        total_calories += parse_number(nutrition['calories'])
        total_protein += parse_number(nutrition['protein'])
        total_carbs += parse_number(nutrition['total_carbohydrate'])
        total_fats += parse_number(nutrition['total_fat'])
        total_price += parse_number(df.loc[df['name'] == ingredient, 'price'].iloc[0])
        in_pantry += 1 if ingredient in pantry_ingredients else 0  
        if df.loc[df['name'] == ingredient, 'category'].iloc[0] == 'Meat & Seafood':
            vegetarian = False
    
    print(total_calories, total_protein, total_carbs, total_fats, vegetarian, total_price)
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
    
    reward = 0.45 * ingredient_score + 0.35 * nutrition_score + 0.20 * cost_score
    return reward


def train(df):
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
            
            
    with open('Q_table_.pickle', 'wb') as handle:
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

if __name__ == "__main__":
    df = pd.read_csv("trader_joes_nov_18.csv")
    df = df[df['nutrition_json'].notna() & df['nutrition_json'].astype(str).str.len().gt(2)]
    if train_flag:
        train(df)
    else:
        Q_table = np.load('Q_table_.pickle', allow_pickle=True)
        
        # change
        goal = sample_goal(df)
        
        print("GOAL")
        print(goal)
        final_ingredients = generate_meal(df, goal, Q_table)
        print(final_ingredients)
        score = calculate_reward(df, final_ingredients, goal["pantry"], goal["target_calories"], goal["target_protein"], goal["vegetarian_diet"], goal["target_carbs"], goal["target_fat"], goal["target_price"])
        print(score)