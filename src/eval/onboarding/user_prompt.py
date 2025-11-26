import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from fuzzywuzzy import fuzz, process
from src.eval.onboarding.structs import Store, DietaryGoal, PantryItem, UserConstraints


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

    def fuzzy_search_ingredient(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """Multi-stage fuzzy search for finding matching ingredients
        Returns: List of (index, name, match_score)
        """
        query_lower = query.lower().strip()
        results = []
    
        # Stage 1: Exact substring match (highest priority)
        for idx, row in self.ingredients_df.iterrows():
            name_lower = row['name'].lower()
            # Check if query is a substring of the name
            if query_lower in name_lower:
                # Score based on how much of the name it matches
                score = (len(query_lower) / len(name_lower)) * 100
                # Boost score if it starts with the query
                if name_lower.startswith(query_lower):
                    score += 50
                results.append((idx, row['full_name'], score))

        # Sort by score and take top results
        results = sorted(results, key=lambda x: x[2], reverse=True)[:top_k*2]

        # Stage 2: If no good matches, try word-based matching
        if not results or results[0][2] < 70:
            query_words = set(query_lower.split())

            for idx, row in self.ingredients_df.iterrows():
                name_words = set(row['name'].lower().split())

                # Calculate word overlap
                common_words = query_words & name_words
                if common_words:
                    # Score based on percentage of query words found
                    word_score = (len(common_words) / len(query_words)) * 100

                    # Boost if important words match (first word, main ingredient)
                    if query_words and name_words:
                        first_query_word = list(query_words)[0]
                        first_name_word = list(name_words)[0]
                        if first_query_word == first_name_word:
                            word_score += 30

                    results.append((idx, row['full_name'], word_score))

        # Stage 3: Fall back to fuzzy matching only if needed
        if not results:
            from fuzzywuzzy import fuzz

            for idx, row in self.ingredients_df.iterrows():
                # Use ratio for better matching
                score = fuzz.ratio(query_lower, row['name'].lower())
                if score >= 60:  # Higher threshold
                    results.append((idx, row['full_name'], score))

        # Remove duplicates and sort
        seen = set()
        final_results = []
        for idx, name, score in sorted(results, key=lambda x: x[2], reverse=True):
            if idx not in seen:
                seen.add(idx)
                final_results.append((idx, name, score))
                if len(final_results) >= top_k:
                    break

        return final_results

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
                        category=self.ingredients_df.iloc[selected_idx]["category"]
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
        print("\nPreferred Store in Boston:")
        print("1. Trader Joe's")
        print("2. Whole Foods")
        choice = input("Select (1 or 2): ").strip()
        if choice in ["1", "2"]:
            self.constraints.preferred_store = [
                Store.TRADER_JOES,
                Store.WHOLE_FOODS,
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
        print(f"Pantry vector shape: {numpy_state['current_pantry'].shape}")
        print(
            f"Non-zero pantry entries: {np.count_nonzero(numpy_state['current_pantry'])}"
        )

        return self.constraints, numpy_state