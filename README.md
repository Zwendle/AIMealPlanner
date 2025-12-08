Welcome to the AI Meal Planner project! Developed by Kritika Agarwal, Claire Hendershot, Sabine Laurence, Aahil Nishad, and Zachary Wen

Our AI Meal Planner is a q-learning based model, which generates new weekly meal plan based on user's ingredient preferences, nutrition goals, pantry 
history, and data from Whole Foods and Trader Jo's. 

Features include:
-- Collecting user preferences using an onboarding interface including current ingredients in pantry, nutrition goals, dietary goals, 
number of meals, budget
-- Evaluation report with total score and score per weight 
-- Generated weekly meal plan and price summary 
-- Saves pantry data and tracks leftovers across weekly runs 

How to install: 
1. Clone the github repository:
git clone https://github.com/Zwendle/AIMealPlanner.git
cd AIMealPlanner

2. Install 3.10+ version of python 

3. Install required packages from requirements.txt
If using pip:
pip install -r requirements.txt

If using any other preferred environment:
<your-environment> install -r requirements.txt

To run:
python src/main.py
