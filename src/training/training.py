import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import pickle
import sys
import os


ALPHA = 0.1       
GAMMA = 0.95   
EPSILON = 0.4
DECAY_RATE = 0.999995   
MAX_STEPS = 50
EPISODES = 10000
train_flag = 'train' in sys.argv
      


def make_state(ingredients, goal, df):
   total_cal = sum(df.loc[df['name_clean'] == i, 'calories'].values[0] for i in ingredients)
   total_prot = sum(df.loc[df['name_clean'] == i, 'protein' ].values[0] for i in ingredients)
   total_carb = sum(df.loc[df['name_clean'] == i, 'carbs'   ].values[0] for i in ingredients)
   total_fat  = sum(df.loc[df['name_clean'] == i, 'fat'     ].values[0] for i in ingredients)
   total_price = sum(df.loc[df['name_clean'] == i, 'cost_per_serving'].values[0] for i in ingredients)


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


   for episode in tqdm(range(EPISODES)):


       num_to_pick = random.randint(5, 8)
       ingredients = random.sample(all_ingredients, num_to_pick)


       goal = sample_goal(df)


       for step in range(MAX_STEPS):


           state = make_state(ingredients, goal, df)
           actions = get_actions(all_ingredients, ingredients)


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


           old_q = Q.get((state, action), 0)
           next_actions = get_actions(all_ingredients, next_ingredients)
           future_q = max([Q.get((next_state, a), 0) for a in next_actions]) if next_actions else 0
           Q[(state, action)] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)


           ingredients = next_ingredients
       EPSILON *= DECAY_RATE
          
   os.makedirs('data', exist_ok=True) # run out of /data instead
   with open('data/Q_table.pickle', 'wb') as handle:
       pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)




def generate_meal(df, goal, Q_table, steps=20):
   """
   Generate a meal using the Q-learning model.
  
   Returns:
       dict: Mapping of ingredient name (name_clean) -> servings_used (default 1.0)
   """
   all_ingredients = df['name_clean'].tolist()
  
   num_to_pick = random.randint(5, 8)
   ingredients = random.sample(all_ingredients, num_to_pick)


   for step in range(steps):
       state = make_state(ingredients, goal, df)
       actions = get_actions(all_ingredients, ingredients)


       if not actions:
           break


       q_vals = [Q_table.get((state, a), 0) for a in actions]
       best_action = actions[int(np.argmax(q_vals))]


       ingredients = apply_action(ingredients, best_action)


   # Return as dict mapping ingredient -> servings_used (default 1.0 per ing)
   return {ing: 1.0 for ing in ingredients}


def load_model(path='data/Q_table.pickle'):
   with open(path, 'rb') as handle:
       return pickle.load(handle)


def model_exists(path='data/Q_table.pickle'):
   return os.path.exists(path)


if __name__ == "__main__":
   df = pd.read_csv("data/ingredients.csv")
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

