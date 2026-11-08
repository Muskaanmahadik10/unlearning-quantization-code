"""B05 answer-only loss evaluator. No model loading or production authorization."""
import hashlib
import json
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def summarize(rows):
    require(bool(rows), 'empty evaluation')
    require(len({r['index'] for r in rows}) == len(rows), 'duplicate sample index')
    for r in rows:
        require(isinstance(r['tokens'], int) and r['tokens'] > 0, 'zero/invalid answer tokens')
        require(math.isfinite(r['loss_sum']) and r['loss_sum'] >= 0, 'invalid loss')
    total = sum(r['tokens'] for r in rows)
    return {'sample_count': len(rows), 'answer_tokens': total,
            'token_mean_nll': math.fsum(r['loss_sum'] for r in rows)/total,
            'sample_mean_nll': math.fsum(r['loss_sum']/r['tokens'] for r in rows)/len(rows)}


def compare_reloads(reference, candidate, tolerance=1e-6):
    # Fixed strict threshold; never estimated from observed protected outcomes.
    require(tolerance == 1e-6, 'unapproved tolerance')
    require(len(reference) == len(candidate) and bool(reference), 'sample count mismatch')
    summarize(reference); summarize(candidate)
    deltas = []
    for a, b in zip(reference, candidate):
        require(all(a[k] == b[k] for k in ('index', 'sample_sha256', 'encoding_sha256', 'tokens')),
                'sample/token identity drift')
        deltas.append(abs(a['loss_sum']/a['tokens']-b['loss_sum']/b['tokens']))
    require(max(deltas) <= tolerance, 'per-sample reload loss drift')
    return {'status': 'PASS', 'max_per_sample_nll_delta': max(deltas), 'tolerance': tolerance}


def answer_loss(torch, logits, labels):
    require(logits.ndim == 3 and labels.ndim == 2 and logits.shape[:2] == labels.shape,
            'logit/label shape')
    shifted = labels[:, 1:].contiguous()
    tokens = int((shifted != -100).sum().item())
    require(tokens > 0, 'zero shifted answer tokens')
    require(bool(torch.isfinite(logits).all()), 'nonfinite logits')
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
        shifted.view(-1), ignore_index=-100, reduction='sum')
    value = float(loss.item())
    require(math.isfinite(value), 'nonfinite loss')
    return value, tokens


def evaluate(model, tokenizer, samples):
    """All 128 frozen samples, batch one, max length 512, no generation/cache."""
    import torch
    from gate2_v4_dataset import encode_answer_only, normalize_question, normalize_answer
    require(len(samples) == 128, 'exactly 128 samples required')
    require(all(t.device.type == 'cuda' for t in list(model.parameters())+list(model.buffers())),
            'CPU/disk offload')
    require(model.config._attn_implementation == 'eager', 'attention implementation')
    model.eval()
    rows = []
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            encoded = encode_answer_only(tokenizer, normalize_question(sample), normalize_answer(sample), 512)
            payload = {k: encoded[k].tolist() for k in ('input_ids', 'attention_mask', 'labels')}
            start = int(encoded['answer_start'])
            require(start > 0 and all(v == -100 for v in payload['labels'][:start]), 'prompt mask')
            require(payload['labels'][start:] == payload['input_ids'][start:], 'answer mask')
            inputs = {k: encoded[k].unsqueeze(0).to('cuda:0') for k in ('input_ids', 'attention_mask')}
            labels = encoded['labels'].unsqueeze(0).to('cuda:0')
            logits = model(**inputs, use_cache=False).logits
            loss, tokens = answer_loss(torch, logits, labels)
            rows.append({'index': i, 'sample_sha256': identity(sample),
                         'encoding_sha256': identity(payload), 'tokens': tokens, 'loss_sum': loss})
            del logits
            if (i+1) % 16 == 0:
                print(f'B05_EVALUATION_SAMPLES={i+1}/128', flush=True)
    return {'rows': rows, 'summary': summarize(rows), 'max_length': 512,
            'metric': 'answer_only_nll', 'official_benchmark_evaluation': False}
