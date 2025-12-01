# Pure data structures - no UI logic
from dataclasses import dataclass, field
from enum import Enum
from typing import List
import numpy as np
import pandas as pd


class Store(Enum):
    TRADER_JOES = 0
    WHOLE_FOODS = 1


class DietaryGoal(Enum):
    HIGH_PROTEIN = 0
    KETO = 1
    LOW_CARB = 2
    VEGETARIAN = 3
    NONE = 4


@dataclass
class PantryItem:
    name: str
    category: str
    servings_available: float
    ingredient_idx: int


@dataclass
class UserConstraints:
    pantry: List[PantryItem] = field(default_factory=list)
    dietary_goal: DietaryGoal = DietaryGoal.NONE
    num_meals: int = 5
    budget_min: float = 30.0
    budget_max: float = 50.0

    def to_numpy_state(self, ingredients_df: pd.DataFrame) -> dict:
        """Convert to numpy for GA evaluation"""
        # Create pantry availability vector (length = total ingredients in DB)
        num_ingredients = len(ingredients_df)
        pantry_vector = np.zeros(num_ingredients, dtype=np.float32)
        
        for item in self.pantry:
            if 0 <= item.ingredient_idx < num_ingredients:
                pantry_vector[item.ingredient_idx] = item.servings_available

        # Constraint parameters
        params = np.array(
            [
                self.num_meals,
                self.budget_min,
                self.budget_max,
            ]
        )

        return {
            "current_pantry": pantry_vector,  # What user has
            "dietary_goal": self.dietary_goal,  # Dietary constraints
            "parameters": params,  # Other constraints
        }
