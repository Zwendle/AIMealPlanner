# Pure data structures re-exported for onboarding / evaluation logic.
# Keep this shim so existing imports remain valid while the canonical
# definitions live under src/training/state_model.py.

from ...training.state_model import (
    Store,
    DietaryGoal,
    PantryItem,
    UserConstraints,
)

__all__ = ["Store", "DietaryGoal", "PantryItem", "UserConstraints"]
