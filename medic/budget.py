"""Persistent conservative reservations, not a provider billing guarantee."""
import json
import math
from decimal import Decimal, ROUND_CEILING


class BudgetExceeded(RuntimeError):
    pass


def micros(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('budget amounts must be finite and nonnegative')
    return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


class BudgetLedger:
    def __init__(self, ledger, approval_id, usd, prices):
        self.ledger, self.approval_id = ledger, approval_id
        self.limit = micros(usd)
        if not self.limit or not approval_id:
            raise ValueError('positive budget and approval ID required')
        payload = {'limit_micros': self.limit, 'prices': prices}
        with ledger.key_lock('budget', approval_id):
            previous = next((r['payload'] for r in ledger.records()
                             if r['kind'] == 'BUDGET_APPROVED' and r['natural_key'] == approval_id), None)
            if previous is not None and previous != payload:
                raise ValueError('approval budget/pricing is pinned; create a new approval')
            ledger.append_once('BUDGET_APPROVED', approval_id, payload)

    def _state(self):
        reserved, settled = {}, {}
        for record in self.ledger.records():
            p = record['payload']
            if p.get('approval_id') != self.approval_id:
                continue
            if record['kind'] == 'BUDGET_RESERVED':
                reserved[p['attempt']] = p['micros']
            elif record['kind'] == 'BUDGET_SETTLED':
                settled[p['attempt']] = p['micros']
        return reserved, settled

    def committed(self):
        reserved, settled = self._state()
        return sum(settled.get(key, amount) for key, amount in reserved.items()) / 1_000_000

    def reserve(self, attempt, usd):
        amount = micros(usd)
        with self.ledger.key_lock('budget', self.approval_id):
            reserved, settled = self._state()
            if any(amount > reserved[key] for key, amount in settled.items() if key in reserved):
                raise BudgetExceeded('prior usage exceeded reservation; pricing review and new approval required')
            if attempt in reserved:
                raise BudgetExceeded('attempt already reserved; inspect uncertain charge before retry')
            if sum(settled.get(k, v) for k, v in reserved.items()) + amount > self.limit:
                raise BudgetExceeded('approved budget exhausted; pending reservations remain charged')
            self.ledger.append_once('BUDGET_RESERVED', self.approval_id + ':' + attempt,
                                    {'approval_id': self.approval_id, 'attempt': attempt, 'micros': amount})

    def settle(self, attempt, usd):
        amount = micros(usd)
        with self.ledger.key_lock('budget', self.approval_id):
            reserved, _ = self._state()
            if attempt not in reserved:
                raise ValueError('unknown reservation')
            self.ledger.append_once('BUDGET_SETTLED', self.approval_id + ':' + attempt,
                                    {'approval_id': self.approval_id, 'attempt': attempt, 'micros': amount})
            if amount > reserved[attempt]:
                raise BudgetExceeded('usage exceeded conservative reservation; inspect pricing before continuing')


def request_bound(registry, provider, prompt, schema, output_tokens, images=(), image_tokens_bound=None):
    if images and (image_tokens_bound is None or image_tokens_bound <= 0):
        raise ValueError('image token bound requires explicit verified configuration')
    # UTF-8 bytes bound text conservatively; schema and framing are counted too.
    tokens = len(prompt.encode('utf-8')) + len(json.dumps(schema).encode('utf-8')) + 4096
    tokens += len(images) * (image_tokens_bound or 0)
    if tokens > 200_000 or output_tokens <= 0:
        raise ValueError('request exceeds validated input/output bounds; split the content')
    price = next(p for p in registry.providers if p.provider == provider)
    return 2 * (tokens * price.input_usd_per_million + output_tokens * price.output_usd_per_million) / 1_000_000
