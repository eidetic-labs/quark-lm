from __future__ import annotations

import random
import unittest

from transformer_char_model_test_support import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
    audit_direct_answer_branch_context_coverage,
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_score,
    branch_diversity_snapshot_target_coverage_diagnostics,
    branch_routing_audit_summary,
    direct_answer_branch_context,
    direct_answer_branch_profile,
    direct_answer_dominant_branch_prediction,
    direct_answer_first_error,
    direct_answer_lesson,
    summarize_branch_context_coverage_gate,
    summarize_branch_diversity_target,
    train_direct_answer_branch_collapse_unlikelihood,
)


class TransformerBranchProfilesTest(unittest.TestCase):
    def test_direct_answer_first_error_targets_greedy_mismatch(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" a.", source="qa:color")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=24,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        repair = direct_answer_first_error(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )

        self.assertIsNotNone(repair)
        _context, target_id, predicted_id, position = repair  # type: ignore[misc]
        self.assertEqual(tokenizer.itos[target_id], " ")
        self.assertEqual(tokenizer.itos[predicted_id], ".")
        self.assertEqual(position, 0)

    def test_direct_answer_branch_profile_summarizes_branch_confusion(self) -> None:
        record = {
            "id": "color",
            "prompt": "q:\na:",
            "target": " a.",
        }
        tokenizer = CharTokenizer.train(record["prompt"] + record["target"] + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=24,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        profile = direct_answer_branch_profile(
            model,
            tokenizer,
            [record],
            branch_position=0,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["count"], 1)
        self.assertEqual(profile["correct"], 0)
        self.assertEqual(profile["skipped"], 0)
        self.assertLess(profile["avg_target_margin"], 0.0)
        self.assertGreater(profile["target_rank"]["avg"], 1.0)
        self.assertEqual(profile["target_rank"]["top1_rate"], 0.0)
        self.assertEqual(profile["target_rank"]["vocab_size"], tokenizer.vocab_size)
        self.assertEqual(profile["target_tokens"][0], {"value": " ", "count": 1})
        self.assertEqual(profile["predicted_tokens"][0], {"value": ".", "count": 1})
        self.assertEqual(profile["confusions"][0], {"value": "' '->'.'", "count": 1})
        self.assertEqual(profile["failed_records"][0]["id"], "color")
        self.assertEqual(profile["failed_records"][0]["target_token"], " ")
        self.assertEqual(profile["failed_records"][0]["predicted_token"], ".")
        self.assertGreaterEqual(profile["failed_records"][0]["target_rank"], 2)
        self.assertEqual(
            profile["failed_records"][0]["top_predictions"][0]["token"],
            ".",
        )

    def test_direct_answer_branch_profile_reports_diversity_collapse(self) -> None:
        records = [
            {"id": "near", "prompt": "q: where?\na:", "target": " near."},
            {"id": "green", "prompt": "q: color?\na:", "target": " green."},
        ]
        tokenizer = CharTokenizer.train(
            "".join(record["prompt"] + record["target"] for record in records)
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=43,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        profile = direct_answer_branch_profile(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["diversity"]["target_unique"], 2)
        self.assertEqual(profile["diversity"]["predicted_unique"], 1)
        self.assertEqual(profile["diversity"]["target_token_coverage"], 0.0)
        self.assertEqual(profile["diversity"]["dominant_predicted_token"], ".")
        self.assertEqual(profile["diversity"]["dominant_predicted_count"], 2)
        self.assertEqual(profile["diversity"]["dominant_predicted_rate"], 1.0)
        self.assertTrue(profile["diversity"]["collapsed"])
        self.assertEqual(
            profile["diversity"]["missing_target_tokens"],
            [{"value": "n", "count": 1}, {"value": "g", "count": 1}],
        )

    def test_branch_diversity_target_marks_collapsed_profiles(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "diversity": {
                        "target_unique": 2,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.5,
                        "dominant_predicted_token": "a",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": True,
                        "missing_target_tokens": [{"value": "b", "count": 1}],
                    }
                },
                "self": {
                    "diversity": {
                        "target_unique": 1,
                        "predicted_unique": 1,
                        "target_token_coverage": 1.0,
                        "dominant_predicted_token": "s",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": False,
                        "missing_target_tokens": [],
                    }
                },
            }
        )

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["multi_target_profiles"], 1)
        self.assertEqual(summary["passed_profiles"], 0)
        self.assertEqual(summary["failed_profiles"], 1)
        self.assertEqual(summary["max_dominant_predicted_rate"], 1.0)
        self.assertEqual(summary["min_target_token_coverage"], 0.5)
        self.assertEqual(summary["blocking_evals"][0]["name"], "qa")
        self.assertTrue(summary["blocking_evals"][0]["collapsed"])
        self.assertEqual(
            summary["root_cause"]["root_cause_hypothesis"],
            "profile_local_prediction_collapse",
        )
        self.assertEqual(
            summary["root_cause"]["mode_counts"]["prediction_collapse"],
            1,
        )

    def test_branch_diversity_target_passes_when_targets_are_covered(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "diversity": {
                        "target_unique": 2,
                        "predicted_unique": 2,
                        "target_token_coverage": 1.0,
                        "dominant_predicted_token": "a",
                        "dominant_predicted_rate": 0.5,
                        "collapsed": False,
                        "missing_target_tokens": [],
                    }
                }
            }
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["multi_target_profiles"], 1)
        self.assertEqual(summary["passed_profiles"], 1)
        self.assertEqual(summary["failed_profiles"], 0)
        self.assertEqual(summary["blocking_evals"], [])
        self.assertEqual(
            summary["root_cause"]["root_cause_hypothesis"],
            "no_branch_diversity_gap",
        )

    def test_branch_diversity_root_cause_detects_global_dominant_token(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "target_rank": {"avg": 14.0, "top3_rate": 0.0, "top5_rate": 0.125},
                    "diversity": {
                        "target_unique": 8,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.125,
                        "dominant_predicted_token": "n",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": True,
                        "missing_target_tokens": [{"value": "a", "count": 3}],
                    },
                },
                "heldout": {
                    "target_rank": {"avg": 13.5, "top3_rate": 0.0, "top5_rate": 0.125},
                    "diversity": {
                        "target_unique": 8,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.125,
                        "dominant_predicted_token": "n",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": True,
                        "missing_target_tokens": [{"value": "b", "count": 2}],
                    },
                },
            }
        )

        root_cause = summary["root_cause"]
        self.assertEqual(
            root_cause["root_cause_hypothesis"],
            "global_output_prior_collapse",
        )
        self.assertEqual(root_cause["severity"], "critical")
        self.assertTrue(root_cause["global_dominant_token_reuse"])
        self.assertEqual(root_cause["collapsed_profile_count"], 2)
        self.assertEqual(
            root_cause["reused_dominant_tokens"],
            [{"token": "n", "profile_count": 2, "profiles": ["heldout", "qa"]}],
        )
        self.assertIn(
            "Audit global logit priors and output-bias escape paths before adding another objective.",
            root_cause["recommendations"],
        )

    def test_branch_diversity_root_cause_detects_wrong_diverse_predictions(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "target_rank": {"avg": 3.0, "top3_rate": 0.5, "top5_rate": 1.0},
                    "diversity": {
                        "target_unique": 2,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.5,
                        "dominant_predicted_token": "a",
                        "dominant_predicted_rate": 0.5,
                        "collapsed": False,
                        "missing_target_tokens": [{"value": "b", "count": 1}],
                    },
                }
            }
        )

        root_cause = summary["root_cause"]
        self.assertEqual(
            root_cause["root_cause_hypothesis"],
            "wrong_diversity_not_target_coverage",
        )
        self.assertEqual(root_cause["wrong_diverse_profile_count"], 1)
        self.assertEqual(
            root_cause["mode_counts"]["wrong_diverse_predictions"],
            1,
        )
        self.assertIn(
            "Require target-aligned coverage, not just more distinct predicted tokens.",
            root_cause["recommendations"],
        )

    def test_branch_routing_audit_flags_output_bias_escape_risk(self) -> None:
        branch_profiles = {
            "qa": {
                "target_tokens": [{"value": "a", "count": 1}, {"value": "b", "count": 1}],
                "predicted_tokens": [{"value": "n", "count": 2}],
                "target_rank": {"avg": 20.0, "top3_rate": 0.0, "top5_rate": 0.0},
                "diversity": {
                    "target_unique": 2,
                    "predicted_unique": 1,
                    "target_token_coverage": 0.0,
                    "dominant_predicted_token": "n",
                    "dominant_predicted_rate": 1.0,
                    "collapsed": True,
                    "missing_target_tokens": [
                        {"value": "a", "count": 1},
                        {"value": "b", "count": 1},
                    ],
                },
            },
            "heldout": {
                "target_tokens": [{"value": "c", "count": 1}, {"value": "d", "count": 1}],
                "predicted_tokens": [{"value": "n", "count": 2}],
                "target_rank": {"avg": 20.0, "top3_rate": 0.0, "top5_rate": 0.0},
                "diversity": {
                    "target_unique": 2,
                    "predicted_unique": 1,
                    "target_token_coverage": 0.0,
                    "dominant_predicted_token": "n",
                    "dominant_predicted_rate": 1.0,
                    "collapsed": True,
                    "missing_target_tokens": [
                        {"value": "c", "count": 1},
                        {"value": "d", "count": 1},
                    ],
                },
            },
        }

        audit = branch_routing_audit_summary(
            branch_profiles,
            output_bias_by_token={"n": 4.0, "a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
            branch_logit_prior_profiles={
                "qa": {
                    "dominant_predicted_token": "n",
                    "dominant_predicted_rate": 1.0,
                    "dominant_token_bias": 4.0,
                    "dominant_token_bias_rank": 1,
                    "dominant_vs_target_decomposition": {
                        "failed_records": {
                            "count": 2,
                            "avg_bias_advantage": 4.0,
                            "avg_hidden_advantage": 0.0,
                            "avg_logit_advantage": 4.0,
                            "bias_share_of_positive_advantage": 1.0,
                            "dominant_logit_win_rate": 1.0,
                            "primary_pressure": "output_bias",
                        }
                    },
                }
            },
        )

        self.assertEqual(audit["root_cause_hypothesis"], "global_output_prior_collapse")
        self.assertEqual(audit["audit_hypothesis"], "output_bias_escape_risk")
        self.assertEqual(audit["output_bias"]["escape_risk"], "high")
        self.assertEqual(
            audit["output_bias"]["dominant_tokens"][0]["token"],
            "n",
        )
        self.assertEqual(
            audit["output_bias"]["dominant_tokens"][0]["bias_rank"],
            1,
        )
        self.assertGreater(audit["output_bias"]["dominant_bias_advantage"], 0.0)
        self.assertEqual(audit["logit_prior"]["bias_driven_profile_count"], 1)
        self.assertEqual(
            audit["logit_prior"]["profiles"][0]["primary_pressure"],
            "output_bias",
        )
        self.assertIn(
            "Compare dominant-token bias ranks against missing target-token bias ranks.",
            audit["next_checks"],
        )
        self.assertIn(
            "Trace output-bias update paths for reused dominant tokens before another repair objective.",
            audit["next_checks"],
        )

    def test_branch_routing_audit_flags_representation_and_imbalance_risk(self) -> None:
        branch_profiles = {
            "qa": {
                "target_tokens": [
                    {"value": "a", "count": 4},
                    {"value": "b", "count": 1},
                    {"value": "c", "count": 1},
                ],
                "predicted_tokens": [{"value": "x", "count": 6}],
                "target_rank": {"avg": 10.0, "top3_rate": 0.2, "top5_rate": 0.2},
                "diversity": {
                    "target_unique": 3,
                    "predicted_unique": 1,
                    "target_token_coverage": 0.0,
                    "dominant_predicted_token": "x",
                    "dominant_predicted_rate": 1.0,
                    "collapsed": True,
                    "missing_target_tokens": [
                        {"value": "a", "count": 4},
                        {"value": "b", "count": 1},
                        {"value": "c", "count": 1},
                    ],
                },
            }
        }
        representation_profiles = {
            "qa": {
                "target_unique": 3,
                "different_target_pairwise_distance": {"avg": 0.001},
                "same_target_pairwise_distance": {"avg": 0.001},
            }
        }

        audit = branch_routing_audit_summary(
            branch_profiles,
            representation_profiles,
            output_bias_by_token={"x": 0.0, "a": 0.0, "b": 0.0, "c": 0.0},
        )

        self.assertEqual(
            audit["audit_hypothesis"],
            "representation_separation_risk",
        )
        self.assertEqual(audit["representation"]["low_separation_profile_count"], 1)
        self.assertEqual(
            audit["target_imbalance"]["high_imbalance_profiles"][0]["profile"],
            "qa",
        )
        self.assertEqual(audit["target_imbalance"]["rare_target_token_count"], 2)
        self.assertIn(
            "Balance candidate construction by profile and target token before another objective screen.",
            audit["next_checks"],
        )

    def test_branch_diversity_snapshot_score_prefers_more_prediction_diversity(self) -> None:
        collapsed = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    }
                }
            },
        }
        cracked = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 0.75,
                    }
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(cracked),
            branch_diversity_snapshot_score(collapsed),
        )

    def test_branch_diversity_snapshot_score_uses_target_rank_tiebreaker(self) -> None:
        buried = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 20.0,
                        "top3_rate": 0.0,
                        "top5_rate": 0.0,
                    },
                }
            },
        }
        lifted = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 8.0,
                        "top3_rate": 0.25,
                        "top5_rate": 0.5,
                    },
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(lifted),
            branch_diversity_snapshot_score(buried),
        )

    def test_branch_diversity_snapshot_score_prefers_rank_over_wrong_diversity(self) -> None:
        wrong_diverse = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 0.75,
                    },
                    "target_rank": {
                        "avg": 14.0,
                        "top3_rate": 0.0,
                        "top5_rate": 0.25,
                    },
                }
            },
        }
        rank_lifted = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 12.0,
                        "top3_rate": 0.25,
                        "top5_rate": 0.25,
                    },
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(rank_lifted),
            branch_diversity_snapshot_score(wrong_diverse),
        )

    def test_branch_diversity_snapshot_coverage_floor_is_profile_wise(self) -> None:
        baseline = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "self": {
                    "diversity": {
                        "target_unique": 1,
                        "target_token_coverage": 1.0,
                    }
                },
            }
        }
        rank_lifted_but_forgetting = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.0,
                    },
                    "target_rank": {
                        "avg": 4.0,
                        "top3_rate": 0.5,
                        "top5_rate": 0.75,
                    },
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.5,
                    },
                    "target_rank": {
                        "avg": 4.0,
                        "top3_rate": 0.5,
                        "top5_rate": 0.75,
                    },
                },
            }
        }
        coverage_preserved = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
            }
        }

        self.assertFalse(
            branch_diversity_snapshot_preserves_target_coverage(
                rank_lifted_but_forgetting,
                baseline,
            )
        )
        self.assertTrue(
            branch_diversity_snapshot_preserves_target_coverage(
                coverage_preserved,
                baseline,
            )
        )
        diagnostics = branch_diversity_snapshot_target_coverage_diagnostics(
            rank_lifted_but_forgetting,
            baseline,
        )
        self.assertFalse(diagnostics["preserved"])
        self.assertEqual(diagnostics["violating_profile_count"], 1)
        self.assertEqual(
            diagnostics["worst_violation"],
            {
                "profile": "qa",
                "baseline_coverage": 0.25,
                "snapshot_coverage": 0.0,
                "deficit": 0.25,
            },
        )

    def test_branch_context_coverage_marks_truncated_semantic_branch(self) -> None:
        record = {
            "id": "place",
            "prompt": "question: where is mia's ball?\nanswer:",
            "target": " under.",
        }
        tokenizer = CharTokenizer.train(record["prompt"] + record["target"] + ANSWER_TERMINATOR)
        narrow = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=3,
                feedforward_dim=5,
                seed=41,
            )
        )
        wide = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=48,
                embedding_dim=3,
                feedforward_dim=5,
                seed=41,
            )
        )

        narrow_audit = audit_direct_answer_branch_context_coverage(
            narrow,
            tokenizer,
            [record],
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        wide_audit = audit_direct_answer_branch_context_coverage(
            wide,
            tokenizer,
            [record],
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(narrow_audit["semantic_records"], 1)
        self.assertEqual(narrow_audit["missing"], 1)
        self.assertIn("intent:place", narrow_audit["missing_records"][0]["missing_features"])
        self.assertEqual(wide_audit["covered"], 1)
        self.assertEqual(wide_audit["missing_records"], [])

    def test_branch_context_coverage_marks_ambiguous_context_collisions(self) -> None:
        records = [
            {"id": "one", "prompt": "q: one\na:", "target": " red."},
            {"id": "two", "prompt": "q: two\na:", "target": " blue."},
        ]
        text = "".join(record["prompt"] + record["target"] for record in records)
        tokenizer = CharTokenizer.train(text + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=3,
                embedding_dim=3,
                feedforward_dim=5,
                seed=42,
            )
        )

        audit = audit_direct_answer_branch_context_coverage(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(audit["count"], 2)
        self.assertEqual(audit["unique_contexts"], 1)
        self.assertEqual(audit["collision_contexts"], 1)
        self.assertEqual(audit["ambiguous_contexts"], 1)
        self.assertEqual(audit["max_context_reuse"], 2)
        self.assertEqual(audit["max_target_options"], 2)
        self.assertEqual(audit["ambiguous_records"][0]["context_text"], "a: ")
        self.assertEqual(
            audit["ambiguous_records"][0]["target_tokens"],
            [{"value": "r", "count": 1}, {"value": "b", "count": 1}],
        )

    def test_branch_context_coverage_gate_summarizes_blockers(self) -> None:
        summary = summarize_branch_context_coverage_gate(
            {
                "qa": {
                    "count": 2,
                    "semantic_records": 2,
                    "covered": 1,
                    "missing": 1,
                    "covered_rate": 0.5,
                    "ambiguous_contexts": 1,
                    "collision_contexts": 1,
                    "skipped": 0,
                },
                "self": {
                    "count": 1,
                    "semantic_records": 1,
                    "covered": 1,
                    "missing": 0,
                    "covered_rate": 1.0,
                    "ambiguous_contexts": 0,
                    "collision_contexts": 0,
                    "skipped": 0,
                },
            }
        )

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["covered"], 2)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["ambiguous_contexts"], 1)
        self.assertEqual(summary["blocking_evals"][0]["name"], "qa")

    def test_dominant_branch_prediction_finds_global_wrong_token(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + green.prompt + green.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=38,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        dominant = direct_answer_dominant_branch_prediction(
            model,
            tokenizer,
            [near, green],
            random.Random(8),
            branch_position=1,
            sample_count=0,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertIsNotNone(dominant)
        predicted_id, count, scored = dominant  # type: ignore[misc]
        self.assertEqual(tokenizer.itos[predicted_id], ".")
        self.assertEqual(count, 2)
        self.assertEqual(scored, 2)

    def test_branch_collapse_repair_penalizes_dominant_wrong_token(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + green.prompt + green.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=39,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            near,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(branch)
        near_context, near_target, _position = branch  # type: ignore[misc]
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        before_wrong = model.predict(near_context)[wrong_id]
        before_target = model.predict(near_context)[near_target]
        rng = random.Random(9)

        for _ in range(32):
            train_direct_answer_branch_collapse_unlikelihood(
                model,
                tokenizer,
                near,
                [near, green],
                lesson,
                rng,
                learning_rate=0.08,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=1,
                sample_count=0,
                terminator=ANSWER_TERMINATOR,
            )

        after_probs = model.predict(near_context)
        self.assertLess(after_probs[wrong_id], before_wrong)
        self.assertGreater(after_probs[near_target], before_target)


if __name__ == "__main__":
    unittest.main()
