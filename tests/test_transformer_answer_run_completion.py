import unittest
from types import SimpleNamespace

from transformer_answer_run_completion import complete_transformer_answer_training_run


class TransformerAnswerRunCompletionTest(unittest.TestCase):
    def test_completion_runs_auxiliary_stages_before_finalization(self) -> None:
        setup = SimpleNamespace(
            examples=["example"],
            training_pool=["example"],
            eval_records={"qa": []},
            eval_candidates={"qa": [" answer"]},
            candidates=[" answer"],
            artifacts="artifacts",
            resume_metadata={"resume": False},
            experiment_intent={"intent": True},
            history_path="history.jsonl",
            lessons_path="lessons.jsonl",
            training_candidates=[" answer"],
            generation_config="generation",
            context_coverage={"qa": {}},
            corpus_hygiene={"hygiene": True},
            hygiene_path="hygiene.json",
            retrieval_memory={"memory": True},
            retrieval_memory_path="memory.json",
            memory_consolidation_plan_path="memory-plan.json",
            replay_mixture={"mixture": True},
            replay_mixture_path="replay-mixture.json",
            sweep_plan={"sweep": True},
            sweep_plan_path="sweep-plan.json",
            training_plan_path="training-plan.json",
            training_recipe={"recipe": True},
            training_recipe_path="recipe.json",
            candidate_quarantine={"quarantine": True},
            candidate_quarantine_path="quarantine.json",
            closed_world_verifier={"passed": True},
            verifier_path="verifier.json",
            constraint_first_path="constraint.json",
            experiment_path="experiment.json",
        )
        calls: list[str] = []

        def train_selector(*args: object) -> dict[str, object]:
            calls.append("selector")
            return {"selector": True}

        def train_generator(*args: object) -> dict[str, object]:
            calls.append("generator")
            return {"generator": True}

        def finalize_run(*args: object) -> dict[str, object]:
            calls.append("finalize")
            return {
                "selector_metrics": args[20],
                "generator_metrics": args[21],
                "replay_mixture": args[27],
                "sweep_plan": args[29],
                "training_plan": args[31],
            }

        result = complete_transformer_answer_training_run(
            args=SimpleNamespace(),
            setup=setup,
            model="model",
            tokenizer="tokenizer",
            optimizer="optimizer",
            training_plan={"plan": True},
            baseline={"baseline": True},
            last_snapshot={"last": True},
            post_direct_candidate_snapshot=None,
            post_direct_candidate_snapshot_skipped=True,
            direct_answer_metrics={"direct": True},
            train_selector=train_selector,
            train_generator=train_generator,
            finalize_run=finalize_run,
        )

        self.assertEqual(calls, ["selector", "generator", "finalize"])
        self.assertEqual(result["selector_metrics"], {"selector": True})
        self.assertEqual(result["generator_metrics"], {"generator": True})
        self.assertEqual(result["replay_mixture"], {"mixture": True})
        self.assertEqual(result["sweep_plan"], {"sweep": True})
        self.assertEqual(result["training_plan"], {"plan": True})


if __name__ == "__main__":
    unittest.main()
