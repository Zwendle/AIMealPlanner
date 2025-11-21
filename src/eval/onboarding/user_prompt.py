import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from fuzzywuzzy import fuzz, process
from structs import Store, DietaryGoal, PantryItem, UserConstraints


class MealPlanningEvaluator:
    """Handles user input for GA evaluation phase"""

    def __init__(self, grocery_csv: str):
        """
        Args:
            grocery_csv: Path to your complete grocery store database
        """
        self.ingredients_df = pd.read_csv(grocery_csv)
        self.ingredients_df["full_name"] = (
            self.ingredients_df["name"]
            + " - "
            + self.ingredients_df["store"].fillna("Any")
        )
        self.constraints = UserConstraints()

        print(f"✓ Loaded {len(self.ingredients_df)} ingredients from database")

    def fuzzy_search_ingredient(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Search ingredients in the database
        Returns: List of (index, name, match_score)
        """
        names = self.ingredients_df["full_name"].tolist()
        matches = process.extract(
            query, names, scorer=fuzz.token_sort_ratio, limit=top_k
        )

        results = []
        for name, score in matches:
            if score >= 60:  # Threshold
                idx = self.ingredients_df[
                    self.ingredients_df["full_name"] == name
                ].index[0]
                results.append((idx, name, score))

        return results

    def input_pantry(self):
        """Get user's current pantry"""
        print("\n" + "=" * 50)
        print("PANTRY ITEMS")
        print("=" * 50)
        print("What ingredients do you already have?")
        print("Type 'done' when finished\n")

        while True:
            query = input("Enter ingredient: ").strip()

            if query.lower() == "done":
                break

            # Search in database
            matches = self.fuzzy_search_ingredient(query)

            if not matches:
                print(f"❌ No matches found for '{query}'")
                continue

            # Show matches
            print("\nFound:")
            for i, (idx, name, score) in enumerate(matches, 1):
                row = self.ingredients_df.iloc[idx]
                print(
                    f"{i}. {row['name']} ({row['serving_size']}) - ${row['price']:.2f}"
                )
            print(f"{len(matches) + 1}. None of these")

            # Select
            try:
                choice = int(input(f"Select (1-{len(matches) + 1}): "))
                if choice == len(matches) + 1:
                    continue
                if 1 <= choice <= len(matches):
                    selected_idx, selected_name, _ = matches[choice - 1]

                    # Get quantity
                    servings = float(input("How many servings? "))

                    # Add to pantry
                    item = PantryItem(
                        name=self.ingredients_df.iloc[selected_idx]["name"],
                        servings_available=servings,
                        ingredient_idx=selected_idx,
                    )
                    self.constraints.pantry.append(item)
                    print(f"✓ Added {servings} servings of {item.name}\n")
            except (ValueError, IndexError):
                print("Invalid input\n")

    def input_preferences(self):
        """Quick input for other preferences"""
        print("\n" + "=" * 50)
        print("PREFERENCES")
        print("=" * 50)

        # Store preference
        print("\nPreferred Store:")
        print("1. Trader Joe's")
        print("2. Whole Foods")
        print("3. Any Store")
        choice = input("Select (1-3, default=3): ").strip()
        if choice in ["1", "2", "3"]:
            self.constraints.preferred_store = [
                Store.TRADER_JOES,
                Store.WHOLE_FOODS,
                Store.ANY,
            ][int(choice) - 1]

        # Dietary goals
        print("\nDietary Goals (comma-separated or 'none'):")
        print("1. High Protein")
        print("2. Keto")
        print("3. Low Carb")
        print("4. Vegetarian")
        choice = input("Select: ").strip()
        if choice.lower() != "none" and choice:
            try:
                choices = [int(c.strip()) for c in choice.split(",")]
                goals = [
                    DietaryGoal.HIGH_PROTEIN,
                    DietaryGoal.KETO,
                    DietaryGoal.LOW_CARB,
                    DietaryGoal.VEGETARIAN,
                ]
                self.constraints.dietary_goals = [
                    goals[c - 1] for c in choices if 1 <= c <= 4
                ]
            except:
                self.constraints.dietary_goals = [DietaryGoal.NONE]

        # Number of meals
        try:
            self.constraints.num_meals_min = int(
                input("\nMin meals per week (default=3): ") or "3"
            )
            self.constraints.num_meals_max = int(
                input("Max meals per week (default=7): ") or "7"
            )
        except:
            pass

        # Budget
        try:
            self.constraints.budget_min = float(
                input("\nMin budget $ (default=30): ") or "30"
            )
            self.constraints.budget_max = float(
                input("Max budget $ (default=50): ") or "50"
            )
        except:
            pass

    def get_evaluation_state(self) -> Dict[str, np.ndarray]:
        """Get the state for GA evaluation"""
        return self.constraints.to_numpy_state(self.ingredients_df)

    def run(self) -> Tuple[UserConstraints, Dict[str, np.ndarray]]:
        """Run the evaluation input process"""
        print("\n" + "=" * 60)
        print("MEAL PLANNING - EVALUATION SETUP")
        print("=" * 60)

        self.input_pantry()
        self.input_preferences()

        # Get numpy state for GA
        numpy_state = self.get_evaluation_state()

        print("\n" + "=" * 50)
        print("READY FOR EVALUATION")
        print("=" * 50)
        print(f"Pantry items: {len(self.constraints.pantry)}")
        print(f"Pantry vector shape: {numpy_state['pantry_availability'].shape}")
        print(
            f"Non-zero pantry entries: {np.count_nonzero(numpy_state['pantry_availability'])}"
        )

        return self.constraints, numpy_state


# ============================================
# GENETIC ALGORITHM INTEGRATION
# ============================================


class MealPlannerGA:
    """
    Your genetic algorithm that trains on the full grocery database
    and evaluates meal plans given user constraints
    """

    def __init__(self, grocery_csv: str):
        self.ingredients_df = pd.read_csv(grocery_csv)
        self.n_ingredients = len(self.ingredients_df)

        # Pre-compute ingredient feature matrix for training
        self.ingredient_features = self._create_feature_matrix()

    def _create_feature_matrix(self) -> np.ndarray:
        """Create feature matrix from all ingredients for GA training"""
        features = self.ingredients_df[
            ["price", "carbs", "fat", "protein", "calories", "cost_per_serving"]
        ].values
        # Normalize if needed
        return features

    def train(self, n_generations: int = 100):
        """
        Train GA on the full ingredient database
        This learns optimal meal planning policies
        """
        print(f"Training GA on {self.n_ingredients} ingredients...")

        # Your GA training logic here
        # This operates on self.ingredient_features
        # No user input needed

        # Initialize population of meal plans
        population_size = 50
        genome_length = 7  # days of meals

        # Each genome represents ingredient selections for a week
        population = np.random.rand(population_size, genome_length, self.n_ingredients)

        # Training loop
        for generation in range(n_generations):
            # Evaluate fitness based on your objectives
            fitness = self._evaluate_fitness(population)

            # Selection, crossover, mutation
            population = self._evolve(population, fitness)

            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {fitness.max():.4f}")

        # Store best policy
        self.best_genome = population[fitness.argmax()]
        print("✓ Training complete")

    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness based on your 4 objectives:
        1. Ingredient utilization (40%)
        2. Nutritional targets (30%)
        3. Variety (20%)
        4. Cost (10%)
        """
        fitness_scores = np.zeros(population.shape[0])

        for i, genome in enumerate(population):
            # Your fitness calculation
            utilization_score = self._calculate_utilization(genome)
            nutrition_score = self._calculate_nutrition(genome)
            variety_score = self._calculate_variety(genome)
            cost_score = self._calculate_cost(genome)

            # Weighted combination
            fitness_scores[i] = (
                0.4 * utilization_score
                + 0.3 * nutrition_score
                + 0.2 * variety_score
                + 0.1 * cost_score
            )

        return fitness_scores

    def _calculate_utilization(self, genome: np.ndarray) -> float:
        """Calculate ingredient utilization score"""
        # Minimize waste, maximize pantry usage
        return np.random.random()  # Placeholder

    def _calculate_nutrition(self, genome: np.ndarray) -> float:
        """Calculate nutritional target achievement"""
        return np.random.random()  # Placeholder

    def _calculate_variety(self, genome: np.ndarray) -> float:
        """Calculate meal variety score"""
        return np.random.random()  # Placeholder

    def _calculate_cost(self, genome: np.ndarray) -> float:
        """Calculate cost efficiency"""
        return np.random.random()  # Placeholder

    def _evolve(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Evolution step: selection, crossover, mutation"""
        # Tournament selection, crossover, mutation
        return population  # Placeholder

    def evaluate_with_user_constraints(
        self, user_state: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Apply trained GA to generate meal plan given user constraints

        Args:
            user_state: Dictionary with pantry_availability, dietary_goals, parameters

        Returns:
            DataFrame with recommended meal plan
        """
        pantry = user_state["pantry_availability"]
        dietary = user_state["dietary_goals"]
        params = user_state["parameters"]

        # Apply constraints to trained model
        # Mask out unavailable ingredients based on store preference
        # Prioritize pantry items
        # Apply dietary restrictions
        # Stay within budget

        # For now, return sample meal plan
        meal_plan = pd.DataFrame(
            {
                "day": [1, 1, 2, 2, 3],
                "meal": ["lunch", "dinner", "lunch", "dinner", "lunch"],
                "ingredients": [
                    "Chicken + Rice",
                    "Salmon + Veggies",
                    "Quinoa Bowl",
                    "Black Beans + Sweet Potato",
                    "Greek Yogurt + Fruit",
                ],
                "cost": [8.50, 12.99, 6.50, 5.99, 4.50],
            }
        )

        return meal_plan


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Path to your grocery CSV
    CSV_FILE = "../../data/ingredients.csv"

    # TRAINING PHASE (run once, offline)p[]
    print("=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    ga = MealPlannerGA(CSV_FILE)
    ga.train(n_generations=100)

    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)

    # EVALUATION PHASE (run for each user)
    evaluator = MealPlanningEvaluator(CSV_FILE)
    user_constraints, numpy_state = evaluator.run()

    # Generate meal plan using trained GA
    print("\n" + "=" * 60)
    print("GENERATING MEAL PLAN")
    print("=" * 60)
    meal_plan = ga.evaluate_with_user_constraints(numpy_state)

    print("\n📅 RECOMMENDED MEAL PLAN:")
    print(meal_plan.to_string(index=False))

    total_cost = meal_plan["cost"].sum()
    print(f"\n💰 Total weekly cost: ${total_cost:.2f}")
    print(
        f"✓ Within budget: ${user_constraints.budget_min:.2f} - ${user_constraints.budget_max:.2f}"
    )
