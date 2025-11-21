import pandas as pd
from src.core.structs import UserConstraints
from src.eval.onboarding import UserPrompt
from eval.evaluation import MealPlanEvaluator
# from src.training.trainer import GATrainer


def main():
    # Load data
    ingredients_df = pd.read_csv("data/ingredients.csv")

    # Training phase (if needed)
    # if not model_exists():
    #     trainer = GATrainer(ingredients_df)
    #     model = trainer.train()
    #     save_model(model)
    # else:
    #     model = load_model()

    # Evaluation phase
    prompter = UserPrompt(ingredients_df)
    user_constraints = prompter.run()

    evaluator = MealPlanEvaluator(model, ingredients_df)
    meal_plan = evaluator.evaluate(user_constraints)

    print(meal_plan)


if __name__ == "__main__":
    main()
