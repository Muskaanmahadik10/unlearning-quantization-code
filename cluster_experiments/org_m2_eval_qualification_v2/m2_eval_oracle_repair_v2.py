"""Independent indexed gather, using the frozen CE class axis/reduction topology.

This does not call CrossEntropyLoss or the upstream probability evaluator. The
old last-axis calculation is retained as diagnostic evidence, never as a score.
"""
import math


def probability_oracle(torch, logits, labels, input_ids):
    if logits.dtype != torch.float32:
        raise ValueError('oracle requires the evaluator FP32 logits')
    if labels.shape != input_ids.shape or logits.shape[:2] != labels.shape:
        raise ValueError('oracle tensor shape')
    mask = labels != -100
    if not bool((labels[:, 0] == -100).all()) or not bool((mask.sum(-1) > 1).all()):
        raise ValueError('oracle first token/answer count')
    if not torch.equal(labels[mask], input_ids[mask]):
        raise ValueError('oracle labels do not match token identities')
    for row in mask:
        indices = row.nonzero(as_tuple=True)[0]
        if int(indices[-1] - indices[0] + 1) != len(indices):
            raise ValueError('oracle discontinuous answer labels')
    # Upstream evaluate_probability uses contiguous [B,T-1,V], then transposes
    # to [B,V,T-1] for CE. Match that FP32 normalization axis, but independently
    # gather labelled tokens, mask, count and reduce without invoking CE/NLLLoss.
    shifted = logits[:, :-1, :].contiguous()
    logp = torch.nn.functional.log_softmax(shifted.transpose(-1, -2), dim=1)
    target = labels[:, 1:]
    selected = logp.gather(1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
    token_nll = (-selected).masked_fill(target == -100, 0.)
    average = token_nll.sum(dim=-1) / mask.sum(dim=-1)
    probability = torch.exp(-average)
    if not bool(torch.isfinite(average).all() and torch.isfinite(probability).all()):
        raise ValueError('oracle nonfinite probability/NLL')
    del logp, selected
    # Preserve exactly the old observer's alternate FP32 path for causal audit.
    old_logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    old_selected = old_logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    old = torch.stack([(-old_selected[i] * mask[i, 1:].to(old_selected.dtype)).sum()
                       / int(mask[i].sum()) for i in range(len(labels))])
    diagnostic = {'method': 'FP32_class_axis_gather_no_CE',
                  'legacy_avg_loss': old.cpu().tolist(),
                  'class_axis_avg_loss': average.cpu().tolist(),
                  'legacy_vs_class_axis_max_delta': float((old-average).abs().max()),
                  'tolerance_unchanged': 1e-6}
    return average, probability, diagnostic


def probability_precheck(torch, model, tokenizer, panels, upstream):
    """Fail on the first three fixed QA rows before the generation-heavy suite."""
    from types import SimpleNamespace
    from m2_real_evaluator_v1 import datasets, collator
    from m2_metric_assembly_v1 import upstream_namespace
    from m2_eval_qualification_oracle_v1 import ObservedModel
    from m2_eval_qualification_common_v1 import compare, require
    observer = ObservedModel(torch, model)
    class View:
        device = model.device
        def __call__(self, **batch):
            return SimpleNamespace(logits=observer(**batch, use_cache=False).logits.float())
    rows = panels['forget']
    with torch.no_grad():
        result = upstream_namespace(upstream)['probability'](View(), data=datasets(rows, 'answer'),
            collators=collator(torch, tokenizer.pad_token_id), batch_size=2)
    observed = [row for call in observer.forward_calls for row in call]
    require(list(result['value_by_index']) == [0, 1, 2] and len(observed) == 3, 'precheck coverage')
    report = {'status':'EARLY_PROBABILITY_PARITY_PASS', 'rows':3,
              'layout_diagnostics':observer.layout_diagnostics, 'no_generation':True,
              'observed':observed, 'upstream':result['value_by_index']}
    try:
        report['max_delta'] = max(compare({k:row[k] for k in ('avg_loss','prob')}, result['value_by_index'][i],
                            path='early_probability/'+str(i)) for i,row in enumerate(observed))
    except ValueError as e:
        report['status']='FAIL';report['error']=str(e)
    return report
