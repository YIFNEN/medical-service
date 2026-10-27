from __future__ import annotations
import tempfile, unittest
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from medic.config import ModelRegistry
from medic.providers import ApprovedGateway, RecordedGateway
from medic.research_contracts import EvidenceCard, QuestionSet
PROVIDERS=("openai","anthropic","google")
def deferred_domains():
    from medic.research_contracts import ANKLE_REVIEW_DOMAINS
    return [{"domain": domain, "status": "DEFERRED", "explanation": "Synthetic fixture has no clinical findings",
             "hypotheses": [], "supporting_fact_ids": [], "evidence_ids": [], "counterevidence": [],
             "uncertainties": ["Synthetic data"], "conditional_actions": [], "decision_changers": ["Clinical records"]}
            for domain in ANKLE_REVIEW_DOMAINS]

def opinion(p): return {"opinion_id":f"op-{p}","provider":p,"model":f"model-{p}","provider_response_id":f"response-op-{p}","hypotheses":[f"hypothesis-{p}"],"evidence_ids":["ev-1"],"missing_information":["weight-bearing radiograph"],"action_ranking":["obtain comparable imaging"],"reasoning_summary":f"independent {p}","claim_abstentions":["cannot infer cartilage loss"],"domain_assessments":deferred_domains()}
def peer_batch(p):
    from medic.tribunal import blind_opinion_id
    targets=[x for x in PROVIDERS if x!=p]
    return {"reviewer_provider":p,"reviewer_model":f"model-{p}","provider_response_id":f"response-peer-{p}","reviews":[{"review_id":f"review-{p}-{t}","reviewer_provider":p,"reviewer_model":f"model-{p}","target_opinion_id":blind_opinion_id(p, f"op-{t}"),"verdict":"PARTIAL_SUPPORT","strengths":["uses evidence"],"omissions":["needs comparable imaging"],"evidence_errors":["none found"],"applicability_errors":["follow-up interval differs"],"required_changes":["state imaging limitation"],"citation_errors":[],"evidence_contradictions":[],"evidence_ids":["ev-1"]} for t in targets]}
def revision(p): return {"revised_opinion_id":f"revised-{p}","original_opinion_id":f"op-{p}","provider":p,"model":f"model-{p}","disposition":"MODIFY","response_to_reviews":["accepted limitation"],"hypotheses":[f"hypothesis-{p}"],"evidence_ids":["ev-1"],"missing_information":["weight-bearing radiograph"],"action_ranking":["obtain comparable imaging"],"reasoning_summary":"revised","domain_assessments":deferred_domains()}
def synthesis(): return {"synthesis_id":"synthesis-1","provider":"openai","model":"model-openai","agreements":["more comparable imaging is needed"],"disagreements":["degree of PTOA"],"conditional_action_ranking":["weight-bearing radiograph first"],"conditional_actions":[{"rank":1,"action":"weight-bearing radiograph first","prerequisites":["weight-bearing is safe"],"benefits":["alignment assessment"],"downsides":["radiation exposure"],"evidence_ids":["ev-1"]}],"decision_changers":["joint-space trend"],"evidence_ids":["ev-1"]}
def claim_audits():
    import json
    from medic.claim_grounding import synthesis_claims
    from medic.research_contracts import ProposedSynthesis
    return [{"claim_id": claim["claim_id"], "verdict": "SUPPORTED", "rationale": "Synthetic support",
             "evidence_ids": ["ev-1"], "patient_fact_ids": [],
             "source_quotes": [{"evidence_id": "ev-1", "quote": "finding"}]}
            for claim in synthesis_claims(ProposedSynthesis.model_validate_json(json.dumps(synthesis())))]

def synthesis_review(p): return {"synthesis_review_id":f"synthesis-review-{p}","reviewer_provider":p,"reviewer_model":f"model-{p}","synthesis_id":"synthesis-1","verdict":"PARTIAL_SUPPORT","supported_points":["conditional ranking"],"objections":["uncertainty remains"],"required_changes":["retain dissent"],"citation_errors":[],"evidence_contradictions":[],"evidence_ids":["ev-1"],"claim_reviews":claim_audits()}
def gateways():
    raw={"openai":RecordedGateway("openai","model-openai",[opinion("openai"),peer_batch("openai"),revision("openai"),synthesis(),synthesis_review("openai")]),"anthropic":RecordedGateway("anthropic","model-anthropic",[opinion("anthropic"),peer_batch("anthropic"),revision("anthropic"),synthesis_review("anthropic")]),"google":RecordedGateway("google","model-google",[opinion("google"),peer_batch("google"),revision("google"),synthesis_review("google")])}
    return raw,{k:ApprovedGateway(v) for k,v in raw.items()}
class TribunalTests(unittest.TestCase):
    def evidence(self): return EvidenceCard(evidence_id="ev-1",source="PUBMED",title="Study",url="https://pubmed.ncbi.nlm.nih.gov/1/",retrieved_at=datetime.now(UTC),study_design="cohort",population="adults",findings=("finding",),limitations=("limitation",))
    def test_full_tribunal_preserves_independence_and_cross_review(self):
        from medic.tribunal import TribunalService
        with tempfile.TemporaryDirectory() as temp:
            raw,approved=gateways(); result=TribunalService(Path(temp),approved).run(case_id="case-1",patient_context={"confirmed_facts":["right ankle ORIF"],"uncertainties":[]},question_set=QuestionSet.for_comprehensive_review("case-1","관절염 가능성?"),evidence_cards=(self.evidence(),),estimate=ModelRegistry.default().estimate_tribunal_cost(1000).approve())
            self.assertEqual(len(result.opinions),3); self.assertEqual(len(result.peer_reviews),6); self.assertEqual(Counter(r.target_opinion_id for r in result.peer_reviews),{"op-openai":2,"op-anthropic":2,"op-google":2}); self.assertEqual(len(result.synthesis_reviews),3)
            for p in PROVIDERS: self.assertNotIn("op-",raw[p].prompts[0])
    def test_resume_does_not_repeat_completed_provider_stage(self):
        from medic.tribunal import TribunalService
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); first=RecordedGateway("openai","model-openai",[opinion("openai")]); failing={"openai":ApprovedGateway(first),"anthropic":ApprovedGateway(RecordedGateway("anthropic","model-anthropic",[])),"google":ApprovedGateway(RecordedGateway("google","model-google",[]))}; estimate=ModelRegistry.default().estimate_tribunal_cost(1000).approve(); questions=QuestionSet.for_comprehensive_review("case-1"); evidence=(self.evidence(),)
            with self.assertRaisesRegex(RuntimeError,"no recorded response"): TribunalService(root,failing).run(case_id="case-1",patient_context={"confirmed_facts":[],"uncertainties":[]},question_set=questions,evidence_cards=evidence,estimate=estimate,run_id="run-resume")
            resumed=RecordedGateway("openai","model-openai",[peer_batch("openai"),revision("openai"),synthesis(),synthesis_review("openai")]); _,normal=gateways(); normal["openai"]=ApprovedGateway(resumed); result=TribunalService(root,normal).run(case_id="case-1",patient_context={"confirmed_facts":[],"uncertainties":[]},question_set=questions,evidence_cards=evidence,estimate=estimate,run_id="run-resume")
            self.assertEqual(result.run_id,"run-resume"); self.assertEqual(resumed.call_count,4); self.assertEqual(len(result.calls),13)
    def test_report_completion_fails_when_synthesis_review_finds_evidence_contradiction(self):
        from medic.tribunal import TribunalService
        with tempfile.TemporaryDirectory() as temp:
            bad_review=synthesis_review("google"); bad_review["evidence_contradictions"]=["synthesis reverses the cited finding"]
            raw={
                "openai":RecordedGateway("openai","model-openai",[opinion("openai"),peer_batch("openai"),revision("openai"),synthesis(),synthesis_review("openai")]),
                "anthropic":RecordedGateway("anthropic","model-anthropic",[opinion("anthropic"),peer_batch("anthropic"),revision("anthropic"),synthesis_review("anthropic")]),
                "google":RecordedGateway("google","model-google",[opinion("google"),peer_batch("google"),revision("google"),bad_review]),
            }
            approved={key:ApprovedGateway(value) for key,value in raw.items()}
            with self.assertRaisesRegex(ValueError,"evidence contradiction"):
                TribunalService(Path(temp),approved).run(
                    case_id="case-1",patient_context={"confirmed_facts":[],"uncertainties":[]},
                    question_set=QuestionSet.for_comprehensive_review("case-1"),
                    evidence_cards=(self.evidence(),),
                    estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve(),
                )
if __name__=="__main__": unittest.main()
