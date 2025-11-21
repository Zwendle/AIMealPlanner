# Enums for categorical data
from dataclasses import dataclass
from enum import Enum


class Store(Enum):
    TRADER_JOES = 0  # Numerical for numpy
    WHOLE_FOODS = 1
    ANY = 2


class DietaryGoal(Enum):
    HIGH_PROTEIN = 0
    KETO = 1
    LOW_CARB = 2
    VEGETARIAN = 3
    NONE = 4


@dataclass
class PantryItem:
    """User's pantry item - just tracks what they have"""

    name: str
    servings_available: float
    category: str
    # Index in the grocery CSV for quick lookup
    ingredient_idx: int
