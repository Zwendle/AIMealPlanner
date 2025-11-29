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
    serving_size_grams: float
    servings_available: float
    cost_per_serving: float
    nutrients: Dict[str, float]
    is_meat: bool = False
    ingredient_idx: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)
