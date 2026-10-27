from pathlib import Path
import tempfile
import unittest

from medic.config import ModelRegistry
from medic.providers import ApprovedGateway, RecordedGateway
from medic.research_contracts import QuestionSet
from medic.tribunal import TribunalService
import tests.test_tribunal as fixtures


class ClaimGroundingTests(unittest.TestCase):
    def test_reviewed_imaging_ids_support_grounding_and_domain_review(self):
        import json
        from medic.claim_grounding import ground_synthesis
        from medic.research_contracts import IndependentOpinion, ProposedSynthesis, SynthesisReview
        patient = {'imaging_reviews': [{'review_id': 'review-1', 'review_status': 'USER_REVIEWED',
            'measurements': [{'fact_id': 'measurement-1', 'value': 5.0, 'unit': 'mm'}],
            'segmentations': []}]}
        opinion = fixtures.opinion('openai')
        opinion['domain_assessments'][0]['supporting_fact_ids'] = ['measurement-1']
        TribunalService._validate_domains(IndependentOpinion.model_validate_json(json.dumps(opinion)),
            QuestionSet.for_comprehensive_review('case-1'), patient)
        review = fixtures.synthesis_review('openai')
        for audit in review['claim_reviews']:
            audit.update(patient_fact_ids=['measurement-1'], evidence_ids=[], source_quotes=[])
        synthesis = ProposedSynthesis.model_validate_json(json.dumps(fixtures.synthesis()))
        parsed = SynthesisReview.model_validate_json(json.dumps(review))
        _, withheld = ground_synthesis(synthesis, (parsed,), (), patient)
        self.assertFalse(withheld)
        patient['imaging_reviews'][0]['review_status'] = 'UNREVIEWED'
        _, withheld = ground_synthesis(synthesis, (parsed,), (), patient)
        self.assertTrue(withheld)

    def test_required_but_unread_fulltext_cannot_support_final_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            _, approved = fixtures.gateways()
            card = fixtures.TribunalTests().evidence().model_copy(update={
                'fulltext_required': True, 'fulltext_status': 'UNAVAILABLE', 'analysis_scope': 'ABSTRACT'})
            result = TribunalService(Path(temp), approved).run(case_id='case-1',
                patient_context={'confirmed_facts': [], 'fact_details': [], 'uncertainties': []},
                question_set=QuestionSet.for_comprehensive_review('case-1'), evidence_cards=(card,),
                estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
            self.assertTrue(result.withheld_claims)
            self.assertTrue(any('full text' in reason for claim in result.withheld_claims for reason in claim.reasons))

    def test_bad_claim_is_withheld_while_other_analysis_completes(self):
        from medic.claim_grounding import synthesis_claims
        from medic.research_contracts import ProposedSynthesis
        import json
        with tempfile.TemporaryDirectory() as temp:
            raw, approved = fixtures.gateways()
            review = fixtures.synthesis_review("google")
            review["claim_reviews"][0]["verdict"] = "CONTRADICTED"
            review["claim_reviews"][0]["rationale"] = "The cited text does not support this claim."
            raw["google"]._responses[-1] = review
            result = TribunalService(Path(temp), approved).run(case_id="case-1",
                patient_context={"confirmed_facts": [], "fact_details": [], "uncertainties": []},
                question_set=QuestionSet.for_comprehensive_review("case-1"),
                evidence_cards=(fixtures.TribunalTests().evidence(),),
                estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
            self.assertEqual(len(result.withheld_claims), 1)
            self.assertNotEqual(result.synthesis.agreements[0], fixtures.synthesis()["agreements"][0])
            self.assertEqual(result.synthesis.disagreements[0], fixtures.synthesis()["disagreements"][0])
            self.assertEqual(result.original_synthesis.agreements[0], fixtures.synthesis()["agreements"][0])
