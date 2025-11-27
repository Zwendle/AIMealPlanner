# import pandas as pd
# import random
# import numpy as np
# from tqdm import tqdm
# import pickle
# import sys
# import os

# ALPHA = 0.1        
# GAMMA = 0.99      
# EPSILON = 0.4 
# DECAY_RATE = 0.999995    
# MAX_STEPS = 50
# EPISODES = 10000
# train_flag = 'train' in sys.argv
        

# def make_state(ingredients, goal, df):
#     """ goal_cal_bucket   = goal['target_calories'] // 150
#     goal_prot_bucket  = goal['target_protein'] // 20
#     goal_carb_bucket  = goal['target_carbs'] // 40
#     goal_fat_bucket   = goal['target_fat'] // 20
#     goal_price_bucket = goal['target_price'] // 4 """

#     total_cal = sum(df.loc[df['name_clean'] == i, 'calories'].values[0] for i in ingredients)
#     cal_bucket = total_cal // 150
#     total_prot = sum(df.loc[df['name_clean'] == i, 'protein'].values[0] for i in ingredients)
#     prot_bucket = total_prot // 20
#     total_carb = sum(df.loc[df['name_clean'] == i, 'carbs'].values[0] for i in ingredients)
#     carb_bucket = total_carb // 40
#     total_fat = sum(df.loc[df['name_clean'] == i, 'fat'].values[0] for i in ingredients)
#     fat_bucket = total_fat // 20
#     total_price = sum(df.loc[df['name_clean'] == i, 'cost_per_serving'].values[0] for i in ingredients)
#     price_bucket = total_price // 4

#     return (
#         cal_bucket,
#         prot_bucket,
#         carb_bucket,
#         fat_bucket,
#         price_bucket,
#         goal['vegetarian_diet'])
#     """ goal_cal_bucket,
#     goal_prot_bucket,
#     goal_carb_bucket,
#     goal_fat_bucket,
#     goal_price_bucket """
    

# # used ChatGPT for this function
# def parse_number(value):
#     if isinstance(value, (int, float)):
#         return value
#     if isinstance(value, str):
#         cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
#         return float(cleaned) if cleaned else 0
#     return 0

# def sample_goal(df):
#     all_ingredients = df['name_clean'].tolist()
    
#     diet_type = random.choice(['high_protein', 'keto', 'low_carb', 'balanced'])
    
#     if diet_type == 'high_protein':
#         cal, prot, carbs, fat = 2000, 120, 200, 70
#     elif diet_type == 'keto':
#         cal, prot, carbs, fat = 2000, 100, 30, 120
#     elif diet_type == 'low_carb':
#         cal, prot, carbs, fat = 2000, 80, 100, 90
#     else:
#         cal, prot, carbs, fat = 2000, 50, 275, 78

#     cal = int(cal * random.uniform(0.9, 1.1))
#     prot = int(prot * random.uniform(0.9, 1.1))

#     num_pantry = random.randint(5, 8) 
#     pantry = random.sample(all_ingredients, num_pantry)
    
#     return {
#         "target_calories": cal // 2,  # Per meal
#         "target_protein": prot // 2,
#         "target_carbs": carbs // 2,
#         "target_fat": fat // 2,
#         "vegetarian_diet": random.random() < 0.2,  # 20% vegetarian chance
#         "target_price": random.randint(5, 10),  # Per meal
#         "pantry": pantry
#     }


# def get_actions(all_ingredients, current):
#     actions = []
#     for i in current:
#         for j in all_ingredients:
#             if j not in current:
#                 actions.append((i, j))
#     return actions


# def apply_action(ingredients, action):
#     out_ing, in_ing = action
#     new_list = ingredients.copy()
#     new_list.remove(out_ing)
#     new_list.append(in_ing)
#     return new_list


# def calculate_reward(df, ingredients, pantry_ingredients, target_calories, target_protein, vegetarian_diet, target_carbs, target_fat, target_price):
#     total_calories = 0
#     total_protein = 0
#     total_carbs = 0
#     total_fats = 0
#     total_price = 0
#     in_pantry = 0
#     vegetarian = True
    
#     for ingredient in ingredients:
#         row = df[df['name_clean'] == ingredient].iloc[0]
        
#         total_calories += parse_number(row.get('calories', 0)) 
#         total_protein += parse_number(row.get('protein', 0)) 
#         total_carbs += parse_number(row.get('carbs', 0)) 
#         total_fats += parse_number(row.get('fat', 0))
#         total_price += parse_number(row.get('cost_per_serving', 0)) 
        
#         in_pantry += 1 if ingredient in pantry_ingredients else 0  
        
#         is_meat = row.get('is_meat', False)
#         if is_meat or row.get('category') == 'Meat & Seafood':
#             vegetarian = False
    
#     ingredient_score = in_pantry / len(ingredients)
#     if vegetarian_diet and not vegetarian:
#         ingredient_score *= 0.3
        
#     def deviation_score(total, target):
#         if target == 0:
#             return 1
#         return np.exp(-abs(total - target) / target)

#     calorie_score = deviation_score(total_calories, target_calories)
#     protein_score  = deviation_score(total_protein, target_protein)
#     carb_score     = deviation_score(total_carbs, target_carbs)
#     fat_score      = deviation_score(total_fats, target_fat)
    
#     nutrition_score = np.mean([calorie_score, protein_score, carb_score, fat_score])
    
#     if target_price == 0:
#         cost_score = 1
#     else:
#         ratio = total_price / target_price
#         cost_score = 1 if ratio <= 1 else np.exp(-(ratio - 1))
    
#     # Weights: Utilization (45%), Nutrition (35%), Cost (20%)
#     reward = 0.45 * ingredient_score + 0.35 * nutrition_score + 0.20 * cost_score
#     return reward


# def train(df):
#     Q = {}    
#     global EPSILON
#     all_ingredients = df['name_clean'].tolist()

#     for episode in tqdm(range(EPISODES)):

#         num_to_pick = random.randint(5, 8)
#         ingredients = random.sample(all_ingredients, num_to_pick)

#         goal = sample_goal(df)

#         for step in range(MAX_STEPS):

#             state = make_state(ingredients, goal, df)
#             actions = get_actions(all_ingredients, ingredients)

#             if random.random() < EPSILON:
#                 action = random.choice(actions)
#             else:
#                 q_vals = [Q.get((state, a), 0) for a in actions]
#                 action = actions[int(np.argmax(q_vals))]

#             next_ingredients = apply_action(ingredients, action)
#             next_state = make_state(next_ingredients, goal, df)

#             reward = calculate_reward(
#                 df,
#                 next_ingredients,
#                 goal["pantry"],
#                 goal["target_calories"],
#                 goal["target_protein"],
#                 goal["vegetarian_diet"],
#                 goal["target_carbs"],
#                 goal["target_fat"],
#                 goal["target_price"]
#             )

#             old_q = Q.get((state, action), 0)
#             next_actions = get_actions(all_ingredients, next_ingredients)
#             future_q = max([Q.get((next_state, a), 0) for a in next_actions]) if next_actions else 0
#             Q[(state, action)] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

#             ingredients = next_ingredients
#         EPSILON *= DECAY_RATE
            
#     os.makedirs('data', exist_ok=True) # run out of /data instead
#     with open('data/Q_table.pickle', 'wb') as handle:
#         pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def generate_meal(df, goal, Q_table, steps=20):
#     all_ingredients = df['name_clean'].tolist()
    
#     num_to_pick = random.randint(5, 8)
#     ingredients = random.sample(all_ingredients, num_to_pick)

#     for step in range(steps):
#         state = make_state(ingredients, goal, df)
#         actions = get_actions(all_ingredients, ingredients)

#         if not actions:
#             break

#         q_vals = [Q_table.get((state, a), 0) for a in actions]
#         best_action = actions[int(np.argmax(q_vals))]

#         ingredients = apply_action(ingredients, best_action)

#     return ingredients

# def load_model(path='data/Q_table.pickle'):
#     with open(path, 'rb') as handle:
#         return pickle.load(handle)

# def model_exists(path='data/Q_table.pickle'):
#     return os.path.exists(path)

# if __name__ == "__main__":
#     df = pd.read_csv("data/ingredients.csv")
#     required_cols = ["carbs", "fat", "protein", "calories", "cost_per_serving"]

#     df = df.dropna(subset=required_cols)

#     for col in required_cols:
#         df = df[df[col].astype(str).str.strip() != ""]
        
#     print("Rows after filtering:", len(df))
    
#     if train_flag:
#         train(df)
#     else:
#         if model_exists('data/Q_table.pickle'):
#             Q_table = load_model('data/Q_table.pickle')
#         else:
#             print("Model not found, training...")
#             train(df)
#             Q_table = Q
        
#         # change
#         goal = sample_goal(df)
        
#         print("GOAL")
#         print(goal)
#         final_ingredients = generate_meal(df, goal, Q_table)
#         print(final_ingredients)
#         score = calculate_reward(df, final_ingredients, goal["pantry"], goal["target_calories"], goal["target_protein"], goal["vegetarian_diet"], goal["target_carbs"], goal["target_fat"], goal["target_price"])
#         print(score)

import pandas as pd
import random
import numpy as np
import os
from typing import List, Tuple, Dict

# ---------------- GA PARAMETERS (good defaults) ----------------
POP_SIZE = 50        # population size
GENERATIONS = 80     # generations per meal optimization
MUTATION_RATE = 0.25 # chance to mutate an individual
INGR_MIN = 5
INGR_MAX = 8
# ---------------------------------------------------------------

def parse_number(value):
    """Robust numeric parser for strings like '12 g', '45', nan, etc."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
        try:
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    return 0.0

def calculate_reward(df: pd.DataFrame,
                     ingredients: List[str],
                     pantry_ingredients: List[str],
                     target_calories: float,
                     target_protein: float,
                     vegetarian_diet: bool,
                     target_carbs: float,
                     target_fat: float,
                     target_price: float) -> float:
    """
    Fitness/reward identical in spirit to your previous reward function,
    but uses df['name'] for lookup to match main.py / evaluation.py.
    """
    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fats = 0.0
    total_price = 0.0
    in_pantry = 0
    vegetarian = True

    for ingredient in ingredients:
        # match by 'name' if available, otherwise try 'name_clean'
        if 'name' in df.columns:
            match = df[df['name'] == ingredient]
        else:
            match = df[df['name_clean'] == ingredient]
        if match.empty:
            # fallback: try substring match (case-insensitive)
            lc = ingredient.lower()
            possible = df[df.apply(lambda r: lc in str(r.get('name','')).lower() or lc in str(r.get('name_clean','')).lower(), axis=1)]
            if not possible.empty:
                row = possible.iloc[0]
            else:
                # unknown ingredient => skip but keep algorithm robust
                continue
        else:
            row = match.iloc[0]

        total_calories += parse_number(row.get('calories', 0))
        total_protein += parse_number(row.get('protein', 0))
        total_carbs += parse_number(row.get('carbs', 0))
        total_fats += parse_number(row.get('fat', 0))
        total_price += parse_number(row.get('cost_per_serving', 0))

        if ingredient in pantry_ingredients:
            in_pantry += 1

        if row.get('is_meat', False) or row.get('category') == 'Meat & Seafood':
            vegetarian = False

    # Utilization (ingredient in pantry)
    ingredient_score = (in_pantry / len(ingredients)) if len(ingredients) > 0 else 0.0
    if vegetarian_diet and not vegetarian:
        ingredient_score *= 0.3

    def deviation_score(total, target):
        if target == 0:
            return 1.0
        # smaller penalty for small deviations; exponential decay
        return float(np.exp(-abs(total - target) / (target + 1e-9)))

    calorie_score = deviation_score(total_calories, target_calories)
    protein_score = deviation_score(total_protein, target_protein)
    carb_score = deviation_score(total_carbs, target_carbs)
    fat_score = deviation_score(total_fats, target_fat)

    nutrition_score = float(np.mean([calorie_score, protein_score, carb_score, fat_score]))

    if target_price == 0:
        cost_score = 1.0
    else:
        ratio = total_price / (target_price + 1e-9)
        cost_score = 1.0 if ratio <= 1.0 else float(np.exp(-(ratio - 1.0)))

    # Weights: Utilization (45%), Nutrition (35%), Cost (20%)
    reward = 0.45 * ingredient_score + 0.35 * nutrition_score + 0.20 * cost_score
    return float(reward)

# ----------------- Genetic Algorithm Core --------------------

def _create_individual(all_ingredients: List[str]) -> List[str]:
    k = random.randint(INGR_MIN, INGR_MAX)
    return random.sample(all_ingredients, k)

def _initialize_population(all_ingredients: List[str]) -> List[List[str]]:
    return [_create_individual(all_ingredients) for _ in range(POP_SIZE)]

def _fitness(individual: List[str], df: pd.DataFrame, goal: Dict) -> float:
    return calculate_reward(
        df,
        individual,
        goal["pantry"],
        goal["target_calories"],
        goal["target_protein"],
        goal["vegetarian_diet"],
        goal["target_carbs"],
        goal["target_fat"],
        goal["target_price"]
    )

def _select_parents(population: List[List[str]], fitnesses: List[float]) -> List[List[str]]:
    # tournament selection (size 2)
    parents = []
    n = len(population)
    for _ in range(n):
        a, b = random.sample(range(n), 2)
        winner = population[a] if fitnesses[a] > fitnesses[b] else population[b]
        parents.append(winner)
    return parents

def _crossover(p1: List[str], p2: List[str]) -> List[str]:
    # simple slice-based crossover then enforce size bounds
    if len(p1) == 0 or len(p2) == 0:
        return _create_individual([*p1, *p2])
    cut1 = random.randint(0, len(p1)-1)
    cut2 = random.randint(0, len(p2)-1)
    child = list(dict.fromkeys(p1[:cut1] + p2[cut2:]))  # preserve order, unique
    # enforce bounds
    while len(child) < INGR_MIN:
        choice = random.choice(p1 + p2)
        if choice not in child:
            child.append(choice)
    if len(child) > INGR_MAX:
        child = random.sample(child, INGR_MAX)
    return child

def _mutate(individual: List[str], all_ingredients: List[str]) -> List[str]:
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(individual)-1)
        replacement = random.choice(all_ingredients)
        # try to avoid duplicates
        tries = 0
        while replacement in individual and tries < 5:
            replacement = random.choice(all_ingredients)
            tries += 1
        individual[idx] = replacement
    return individual

def genetic_optimize(df: pd.DataFrame, goal: Dict) -> Tuple[List[str], float]:
    """
    Run GA and return best meal (list of ingredient names) and its fitness.
    """
    if 'name' in df.columns:
        all_ingredients = df['name'].dropna().astype(str).tolist()
    elif 'name_clean' in df.columns:
        all_ingredients = df['name_clean'].dropna().astype(str).tolist()
    else:
        raise ValueError("DataFrame must contain 'name' or 'name_clean' column.")

    population = _initialize_population(all_ingredients)

    for gen in range(GENERATIONS):
        fitnesses = [_fitness(ind, df, goal) for ind in population]
        parents = _select_parents(population, fitnesses)
        if gen % 10 == 0:
            print(f"[GA] Generation {gen:02d} best={max(fitnesses):.4f}")


        children = []
        # create offspring
        for i in range(0, POP_SIZE, 2):
            p1 = parents[i]
            p2 = parents[i+1] if i+1 < len(parents) else parents[0]
            c1 = _crossover(p1, p2)
            c2 = _crossover(p2, p1)
            children.append(_mutate(c1, all_ingredients))
            children.append(_mutate(c2, all_ingredients))

        # elitism: keep top 5 from previous generation
        elite_count = min(5, len(population))
        elite_idx = np.argsort(fitnesses)[-elite_count:]
        elites = [population[i] for i in elite_idx]

        population = children + elites
        # ensure population size
        if len(population) > POP_SIZE:
            population = population[:POP_SIZE]
        # optional small random injection to avoid premature convergence
        if gen % 20 == 0 and gen > 0:
            population[-3:] = [_create_individual(all_ingredients) for _ in range(3)]

    final_fit = [_fitness(ind, df, goal) for ind in population]
    best_idx = int(np.argmax(final_fit))
    return population[best_idx], float(final_fit[best_idx])

# -------------- Backwards-compatible API --------------
def train(df):
    """
    No-op training function kept for compatibility with main.py imports.
    GA performs optimization at inference time, so there's nothing to persist.
    """
    print("GA-based pipeline: no training required. 'train' is a no-op.")

def generate_meal(df: pd.DataFrame, goal: Dict, _model=None) -> List[str]:
    """
    Wrapper used by main.py. The `_model` argument is accepted for compatibility
    but ignored. Returns a list of ingredient names.
    """
    best, score = genetic_optimize(df, goal)
    return best

def load_model(path='data/Q_table.pickle'):
    """Compatibility stub - no model to load."""
    raise FileNotFoundError("GA pipeline does not use persisted Q-table models.")

def model_exists(path='data/Q_table.pickle'):
    """Compatibility stub - always False for GA pipeline."""
    return False
