import unittest

import training_recipe
from constraint_first_report import (
    build_constraint_first_promotion_report,
    promotion_check,
    write_constraint_first_report,
)
from self_improvement_constraints import self_improvement_constraint_report
from training_recipe_core import build_training_recipe, write_training_recipe
from transformer_constraints import transformer_constraint_report


class TrainingRecipeExportsTest(unittest.TestCase):
    def test_compatibility_module_reexports_focused_recipe_apis(self) -> None:
        self.assertIs(training_recipe.build_training_recipe, build_training_recipe)
        self.assertIs(training_recipe.write_training_recipe, write_training_recipe)
        self.assertIs(
            training_recipe.build_constraint_first_promotion_report,
            build_constraint_first_promotion_report,
        )
        self.assertIs(training_recipe.promotion_check, promotion_check)
        self.assertIs(
            training_recipe.write_constraint_first_report,
            write_constraint_first_report,
        )
        self.assertIs(
            training_recipe.self_improvement_constraint_report,
            self_improvement_constraint_report,
        )
        self.assertIs(
            training_recipe.transformer_constraint_report,
            transformer_constraint_report,
        )


if __name__ == "__main__":
    unittest.main()
