"""Outcome-independent condition and transport contracts; no GPU operations."""
from org_b05_behavior_evaluator_v1 import require, summarize

CONDITIONS = ('bf16', 'full_packed', 'safe_looking_singleton',
              'risky_looking_singleton', 'pair', 'abstention')


def condition_sets(cell, targets):
    conditions = cell['conditions']
    require(set(conditions) == set(CONDITIONS), 'six-condition universe')
    require(len(targets) == len(set(targets)) == 224, 'target universe')
    require(conditions['bf16']['protected_layers'] == 'ALL_TARGETS', 'BF16 universe')
    require(conditions['full_packed']['protected_layers'] == [] and
            conditions['full_packed']['alias_of'] is None, 'full packing')
    require(conditions['abstention']['protected_layers'] == [] and
            conditions['abstention']['alias_of'] == 'full_packed', 'abstention identity')
    result = {}
    for name in CONDITIONS:
        row = conditions[name]
        layers = targets if name == 'bf16' else [n.removesuffix('.weight') for n in row['protected_layers']]
        require(len(layers) == len(set(layers)) and set(layers) <= set(targets), 'protected universe')
        if name != 'bf16': require(len(layers) <= 2, 'protected cardinality')
        alias = row.get('alias_of')
        require(alias is None or (alias == 'full_packed' and not layers and
                name in ('abstention','safe_looking_singleton')), 'invalid alias')
        result[name] = {'protected': layers, 'alias_of': alias}
    return result


def transport(packed, emulated):
    """Contrasts relative to full packing. Fixed 1e-6 NLL negligibility floor.

    This is a proposed prospective analysis contract, not a scientific PASS
    merely because serialization worked. A zero eligible count is inconclusive.
    """
    require(set(packed) == set(emulated) == set(CONDITIONS), 'comparison universe')
    rows = []
    for role in ('forget', 'retain'):
        for condition in ('safe_looking_singleton', 'risky_looking_singleton', 'pair'):
            contrasts = []
            for values in (packed, emulated):
                a = values[condition][role]['rows']; b = values['full_packed'][role]['rows']
                require(len(a) == len(b) == 128, 'comparison sample count')
                require(all(all(x[k] == y[k] for k in ('index','sample_sha256','encoding_sha256','tokens'))
                            for x,y in zip(a,b)), 'contrast token identity')
                contrasts.append(summarize(a)['token_mean_nll']-summarize(b)['token_mean_nll'])
            pa = packed[condition][role]['rows']; em = emulated[condition][role]['rows']
            require(all(all(x[k] == y[k] for k in ('index','sample_sha256','encoding_sha256','tokens'))
                        for x,y in zip(pa,em)), 'packed/emulated token identity')
            p,e = contrasts
            eligible = max(abs(p),abs(e)) > 1e-6
            agree = eligible and abs(p)>1e-6 and abs(e)>1e-6 and (p>0)==(e>0)
            rows.append({'condition':condition, 'role':role, 'packed_delta_nll':p,
                         'emulated_delta_nll':e, 'eligible':eligible, 'sign_agreement':agree})
    n = sum(r['eligible'] for r in rows); good = sum(r['sign_agreement'] for r in rows)
    return {'status': 'INCONCLUSIVE' if not n else ('PASS' if good/n >= .9 else 'FAIL'),
            'eligible':n, 'agreements':good, 'fraction':good/n if n else None,
            'non_negligible_nll':1e-6, 'required_fraction':.9, 'rows':rows,
            'production_authorized':False}
