from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class Store(Enum):
    ANY = -1
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
    """
    Canonical definition for a single ingredient the user can cook with.
    All numeric values are expressed per serving.
    """

    name: str
    category: str
    preferred_store: Store
    serving_size_grams: float
    servings_available: float
    cost_per_serving: float
    nutrients: Dict[str, float]
    is_meat: bool = False
    ingredient_idx: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Pantry:
    """
    Container for all available pantry items and helper aggregations.
    """

    items: Dict[str, PantryItem]

    def get(self, name: str) -> Optional[PantryItem]:
        return self.items.get(name)

    def total_cost(self) -> float:
        return sum(item.cost_per_serving * item.servings_available for item in self.items.values())


@dataclass
class DietaryGoals:
    """
    Target-level requirements for the planner.
    All values are optional to allow open-ended plans.
    """

    meals_to_make: int
    budget: Optional[float] = None
    target_macros: Optional[Dict[str, float]] = None
    calorie_limit: Optional[float] = None
    forbidden_categories: List[str] = field(default_factory=list)
    forbidden_ingredients: List[str] = field(default_factory=list)
    required_stores: List[str] = field(default_factory=list)
    preferred_store: Store = Store.ANY
    dietary_focus: List[DietaryGoal] = field(default_factory=list)

    @classmethod
    def from_constraints(cls, constraints: "UserConstraints") -> "DietaryGoals":
        """
        Create a DietaryGoals snapshot from user onboarding constraints.
        Uses the lower bound for meals and budget unless tighter info is available.
        """
        budget_guess = (constraints.budget_min + constraints.budget_max) / 2 if constraints.budget_max else constraints.budget_min
        meals_guess = constraints.num_meals_min
        return cls(
            meals_to_make=meals_guess,
            budget=budget_guess,
            forbidden_categories=[],
            forbidden_ingredients=[],
            required_stores=[constraints.preferred_store.name] if constraints.preferred_store != Store.ANY else [],
            preferred_store=constraints.preferred_store,
            dietary_focus=constraints.dietary_goals,
        )


@dataclass
class UserConstraints:
    """
    User-provided onboarding info that feeds training/eval.
    Shared between onboarding (eval) and training subsystems.
    """

    pantry: List[PantryItem] = field(default_factory=list)
    preferred_store: Store = Store.ANY
    dietary_goals: List[DietaryGoal] = field(default_factory=list)
    num_meals_min: int = 3
    num_meals_max: int = 7
    budget_min: float = 30.0
    budget_max: float = 50.0

    def to_numpy_state(self, ingredients_df: Any) -> Dict[str, Any]:
        """
        Convert the constraint snapshot into numpy-friendly arrays for GA / RL.
        `ingredients_df` is expected to have the same ordering used for ingredient_idx.
        """
        import numpy as np

        ingredient_count = len(ingredients_df) if ingredients_df is not None else max(
            (item.ingredient_idx or 0) for item in self.pantry
        ) + 1
        pantry_vector = np.zeros(ingredient_count, dtype=np.float32)
        for item in self.pantry:
            if item.ingredient_idx is None:
                continue
            pantry_vector[item.ingredient_idx] = np.float32(item.servings_available)

        dietary_vector = np.zeros(len(DietaryGoal), dtype=np.float32)
        for goal in self.dietary_goals:
            dietary_vector[goal.value] = 1.0

        params = np.array(
            [
                self.preferred_store.value,
                self.num_meals_min,
                self.num_meals_max,
                self.budget_min,
                self.budget_max,
            ],
            dtype=np.float32,
        )

        return {
            "current_pantry": pantry_vector,
            "dietary_goals": dietary_vector,
            "parameters": params,
        }

    def to_dietary_goals(self) -> DietaryGoals:
        """Helper to build a DietaryGoals instance for training loops."""
        return DietaryGoals.from_constraints(self)


@dataclass
class MealIngredientPortion:
    """
    Represents the quantity of a specific ingredient used in a meal.
    """

    ingredient_name: str
    servings_used: float


@dataclass
class Meal:
    """
    A candidate meal consisting of 5-8 ingredients plus derived metadata.
    """

    name: str
    portions: List[MealIngredientPortion]
    total_macros: Dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    total_calories: float = 0.0
    score: Optional[float] = None


@dataclass
class MealPlanState:
    """
    Full agent state snapshot used by GA/Q-learning.
    """

    pantry: Pantry
    goals: DietaryGoals
    meals: List[Meal]
    episode_metadata: Dict[str, float] = field(default_factory=dict)

    def to_matrix(self) -> List[List[float]]:
        """
        Optional dense representation:
        M rows (meals) x N columns (pantry items) with servings used.
        """
        item_index = {name: idx for idx, name in enumerate(self.pantry.items.keys())}
        matrix: List[List[float]] = [[0.0 for _ in item_index] for _ in self.meals]
        for meal_idx, meal in enumerate(self.meals):
            for portion in meal.portions:
                col = item_index.get(portion.ingredient_name)
                if col is not None:
                    matrix[meal_idx][col] = portion.servings_used
        return matrix


