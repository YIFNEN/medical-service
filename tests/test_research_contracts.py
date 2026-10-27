from __future__ import annotations

import unittest
from datetime import UTC, datetime

from pydantic import ValidationError


class ResearchContractTests(unittest.TestCase):
    def test_question_set_always_contains_seven_domains_and_custom_question(self):
        from medic.research_contracts import QuestionSet

        questions = QuestionSet.for_comprehensive_review(
            case_id="case-1", custom_question="수술 후 관절염 위험은?"
        )

        self.assertEqual(len(questions.domains), 7)
        self.assertEqual(questions.custom_question, "수술 후 관절염 위험은?")
        self.assertIn("POST_TRAUMATIC_ANKLE_ARTHRITIS", questions.domains)

    def test_direct_question_set_validation_enforces_fixed_domains_and_id(self):
        from medic.research_contracts import QuestionSet

        questions = QuestionSet.for_comprehensive_review("case-1", "question")
        with self.assertRaises(ValidationError):
            QuestionSet.model_validate({
                **questions.model_dump(),
                "domains": tuple(reversed(questions.domains)),
            })
        with self.assertRaises(ValidationError):
            QuestionSet.model_validate({
                **questions.model_dump(),
                "question_set_id": "question-set-forged",
            })

    def test_synthesis_requires_contiguous_evidence_backed_matching_actions(self):
        from medic.research_contracts import ProposedSynthesis

        base = {
            "synthesis_id": "s-1",
            "provider": "openai",
            "model": "model",
            "agreements": (),
            "disagreements": (),
            "conditional_action_ranking": ("first", "second"),
            "conditional_actions": (
                {"rank": 1, "action": "first", "prerequisites": (), "benefits": (),
                 "downsides": (), "evidence_ids": ("ev-1",)},
                {"rank": 2, "action": "second", "prerequisites": (), "benefits": (),
                 "downsides": (), "evidence_ids": ("ev-2",)},
            ),
            "decision_changers": (),
            "evidence_ids": ("ev-1", "ev-2"),
        }
        ProposedSynthesis.model_validate(base)
        invalid_actions = (
            (),
            ({**base["conditional_actions"][0], "rank": 2},),
            (
                {**base["conditional_actions"][0], "rank": 1},
                {**base["conditional_actions"][1], "rank": 1},
            ),
            (
                {**base["conditional_actions"][0], "action": "different"},
                base["conditional_actions"][1],
            ),
            (
                {**base["conditional_actions"][0], "evidence_ids": ()},
                base["conditional_actions"][1],
            ),
        )
        for actions in invalid_actions:
            with self.subTest(actions=actions), self.assertRaises(ValidationError):
                ProposedSynthesis.model_validate({**base, "conditional_actions": actions})

    def test_fact_locators_and_cost_numbers_are_strict(self):
        from medic.config import CostLine
        from medic.research_contracts import ConfirmedFactDetail

        with self.assertRaises(ValidationError):
            ConfirmedFactDetail(
                fact_key="laterality", value="RIGHT", source_locators=(),
                user_confirmed=True,
            )
        with self.assertRaises(ValidationError):
            CostLine(
                stage="analyst", provider="openai", calls=1,
                max_output_tokens=10, estimated_usd=float("inf"),
            )
    def test_opinion_and_evidence_contracts_are_strict_and_immutable(self):
        from medic.research_contracts import EvidenceCard, IndependentOpinion

        card = EvidenceCard(
            evidence_id="ev-1",
            source="PUBMED",
            title="Study",
            url="https://pubmed.ncbi.nlm.nih.gov/1/",
            retrieved_at=datetime.now(UTC),
            study_design="cohort",
            population="adults",
            findings=("finding",),
            limitations=("limitation",),
        )
        opinion = IndependentOpinion(
            opinion_id="op-1",
            provider="openai",
            model="model-1",
            provider_response_id="response-1",
            hypotheses=("PTOA",),
            evidence_ids=(card.evidence_id,),
            missing_information=("weight-bearing radiograph",),
            action_ranking=("obtain comparable imaging",),
            reasoning_summary="source-grounded analysis",
        )

        with self.assertRaises(ValidationError):
            EvidenceCard.model_validate({**card.model_dump(), "unexpected": True})
        with self.assertRaises(ValidationError):
            opinion.provider = "changed"

    def test_cost_estimate_requires_explicit_approval_before_execution(self):
        from medic.config import ModelRegistry

        registry = ModelRegistry.default()
        estimate = registry.estimate_tribunal_cost(input_tokens_per_call=10_000)

        self.assertFalse(estimate.approved)
        self.assertGreater(estimate.estimated_usd, 0)
        self.assertEqual(estimate.call_count, 13)
        self.assertEqual(registry.estimate_extraction_cost(10_000).call_count, 3)
        self.assertEqual(registry.estimate_deliberation_cost(10_000).call_count, 16)
        self.assertEqual(registry.estimate_full_workflow_cost(10_000).call_count, 19)


if __name__ == "__main__":
    unittest.main()
