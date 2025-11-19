import pandas as pd
import random
    
def train(df):
    all_ingredients = df['name'].tolist()

    num_to_pick = random.randint(5, 8)
    ingredients = random.sample(all_ingredients, num_to_pick)
    print(ingredients)
    #get some reward from the combination (based on price, nutritional value, etc)
    #save reward to Q-table for given combo of ingredients (state)
    #swap an ingredient out and get a new reward- save to table
    #continuing swapping NEED TO DETERMINE "DONE" STATE
    #use Q-learning algorithm with decay rate, etc.
    
    

if __name__ == "__main__":
    # based on user input choose between tj/wf
    df = pd.read_csv('trader_joes_nov11.csv')
    train(df)