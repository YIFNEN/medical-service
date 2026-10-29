import pytest

from medic.call_cache import cached_generate
from medic.config import ModelRegistry
from medic.ledger import AppendOnlyJsonlLedger
from medic.provider_checks import SmokeReply
from medic.providers import ApprovedGateway, RawProviderResponse, _LiveGateway


class LocalLiveProbe(_LiveGateway):
    provider = 'openai'
    calls = 0

    def generate_raw(self, **kwargs):
        self.calls += 1
        return RawProviderResponse({'echo': 'MEDIC_SYNTHETIC_SMOKE', 'structured': True}, 'local-response', 1, 1)


def test_personal_live_call_requires_separate_transmission_before_reservation(tmp_path):
    raw = LocalLiveProbe('configured-model', 'UNUSED')
    gateway = ApprovedGateway(raw)
    ledger = AppendOnlyJsonlLedger(tmp_path / 'ledger.jsonl')
    estimate = ModelRegistry.default().estimate_tribunal_cost(100).approve().model_copy(update={'budget_usd': 10.0})
    args = dict(stage='TEST', prompt='synthetic probe', output_model=SmokeReply, estimate=estimate, max_output_tokens=100)
    with pytest.raises(PermissionError, match='transmission'):
        cached_generate(ledger, gateway, scope='test', **args)
    assert raw.calls == 0
    assert not any(r['kind'] == 'BUDGET_RESERVED' for r in ledger.records())
    args['estimate'] = estimate.model_copy(update={'transmission_approved': True})
    assert cached_generate(ledger, gateway, scope='test', **args).value.structured
    assert raw.calls == 1
