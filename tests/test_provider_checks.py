from pathlib import Path

from medic.config import ModelRegistry
from medic.providers import ApprovedGateway, RecordedGateway


def test_smoke_requires_approval_and_resumes_partial_without_repeating(tmp_path):
    from medic.provider_checks import smoke
    import pytest
    registry = ModelRegistry.default()
    raw = {p.provider: RecordedGateway(p.provider, p.model, [{'echo': 'MEDIC_SYNTHETIC_SMOKE', 'structured': True}])
           for p in registry.providers}
    gateways = {name: ApprovedGateway(g) for name, g in raw.items()}
    estimate = registry.estimate_tribunal_cost(100)
    with pytest.raises(PermissionError):
        smoke(tmp_path, 'smoke-test', gateways, registry, estimate)
    assert not any(g.call_count for g in raw.values())
    raw['google']._responses.clear()
    result = smoke(tmp_path, 'smoke-test', gateways, registry, estimate.approve())
    assert result['status'] == 'INCOMPLETE'
    assert result['providers']['google']['status'] == 'FAILED'
    raw['google']._responses.append({'echo': 'MEDIC_SYNTHETIC_SMOKE', 'structured': True})
    result = smoke(tmp_path, 'smoke-test', gateways, registry, estimate.approve())
    assert result['status'] == 'COMPLETE'
    assert [g.call_count for g in raw.values()] == [1, 1, 1]
    assert all(item['response_id'] for item in result['providers'].values())


def test_smoke_does_not_accept_wrong_echo_or_missing_provider(tmp_path):
    from medic.provider_checks import smoke
    import pytest
    registry = ModelRegistry.default()
    with pytest.raises(ValueError, match='three'):
        smoke(tmp_path, 'bad', {}, registry, registry.estimate_tribunal_cost(100).approve())


def test_cli_smoke_without_approval_estimates_only(tmp_path, capsys):
    import json
    from medic.cli import main
    assert main(['providers', 'smoke', '--run-id', 'smoke-cli', '--store', str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['call_count'] == 3
    assert result['paid_calls_made'] == 0
    assert result['input_scope'] == 'SYNTHETIC_ONLY'
