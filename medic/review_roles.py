"""Stable identities and prompts for the three independent review roles."""

from typing import Literal

ReviewRole = Literal['records', 'counterevidence', 'applicability']

REVIEW_ROLES = {
    'records': ('기록·계측 분석',
                'Prioritize reviewed observations and measurements. Explain what each supports and what is missing.'),
    'counterevidence': ('반증·대안 검토',
                        'Look for rival explanations and disconfirming observations. State what would change your view.'),
    'applicability': ('문헌·환자 적용성 검토',
                      'Check whether the cited results entail the claim and apply to this patient and acquisition conditions.'),
}


def review_label(provider, role_id):
    if role_id is None:
        return provider
    return f'{REVIEW_ROLES[role_id][0]} ({provider})'
