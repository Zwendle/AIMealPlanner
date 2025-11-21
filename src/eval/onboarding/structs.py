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
    preferred_store: Store = Store.TRADER_JOES
    dietary_goals: List[DietaryGoal] = field(default_factory=list)
    num_meals_min: int = 3
    num_meals_max: int = 7
    budget_min: float = 30.0
    budget_max: float = 50.0

    def to_numpy_state(self, ingredients_df: pd.DataFrame) -> dict:
        """Convert to numpy for GA evaluation"""
        # Create pantry availability vector (length = total ingredients in DB)
        # Use ingredient name -> servings mapping as a NumPy-friendly dict
        pantry_vector = {
            item.ingredient_idx: np.float32(item.servings_available)
            for item in self.pantry
        }

        # One-hot encode dietary goals
        dietary_vector = np.zeros(5)  # 5 dietary goal options
        for goal in self.dietary_goals:
            dietary_vector[goal.value] = 1

        # Constraint parameters
        params = np.array(
            [
                self.preferred_store.value,
                self.num_meals_min,
                self.num_meals_max,
                self.budget_min,
                self.budget_max,
            ]
        )

        return {
            "current_pantry": pantry_vector,  # What user has
            "dietary_goals": dietary_vector,  # Dietary constraints
            "parameters": params,  # Other constraints
        }
