"""Controlled O/R/U training components. No model loading or submission entrypoint.

The CPU preflight exercises the actual objective/update/resume implementation on
tiny synthetic models. Full-model optimizer, CUDA and checkpoint-loader parity
still require the separately released pilot.
"""
import hashlib
import json
import math
from pathlib import Path
import random


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def schedule(role, config):
    require(role in ('O', 'R', 'U'), 'invalid training role')
    n, epochs, seed = config['epoch_anchor_records'], config['epochs'], config['seed']
    b, acc = config['microbatch'], config['gradient_accumulation']
    require(all(type(x) is int and x > 0 for x in (n, epochs, b, acc)), 'invalid schedule')
    require(n % b == 0, 'partial microbatch needs a separately reviewed weighting policy')
    updates = []
    # Versioned sampling implementation: no claimed bitwise replay of Trainer RNG.
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        pairs = random.Random(seed + 1_000_000 + epoch)
        micros = []
        for start in range(0, n, b):
            ids = order[start:start + b]
            entry = {'anchor': ids}
            if role == 'U':
                entry['retain'] = [pairs.randrange(3600) for _ in ids]
            micros.append(entry)
        for i in range(0, len(micros), acc):
            updates.append({'epoch': epoch, 'microbatches': micros[i:i + acc]})
    require(len(updates) == config['intended_optimizer_steps'], 'optimizer step count drift')
    return {'role': role, 'algorithm': 'python_mt19937_epoch_seed_v1', 'updates': updates,
            'sha256': canonical_sha(updates), 'sample_count': n * epochs,
            'warmup_updates': math.ceil(n / b / acc), 'total_updates': len(updates)}


def lr_multiplier(update_index, total_updates, warmup_updates):
    require(type(update_index) is int and 0 <= update_index < total_updates, 'invalid update cursor')
    require(0 < warmup_updates < total_updates, 'invalid warmup')
    # Matches LambdaLR/linear-warmup convention: first step has learning rate zero.
    if update_index < warmup_updates:
        return update_index / warmup_updates
    return max(0.0, (total_updates - update_index) / (total_updates - warmup_updates))


def encode_qa(tokenizer, question, answer, cap=512):
    require(isinstance(question, str) and isinstance(answer, str) and question and answer, 'empty QA')
    chat = [{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
    ids = list(tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False,
                                             date_string='10 Apr 2025'))
    prompt = list(tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True,
                                                date_string='10 Apr 2025'))
    require(tokenizer.eos_token_id is not None and ids, 'missing EOS or tokens')
    if ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    require(ids[:len(prompt)] == prompt, 'prompt is not exact token prefix')
    require(len(prompt) < len(ids) <= cap, 'length cap or zero answer violation')
    labels = [-100] * len(prompt) + ids[len(prompt):]
    return {'input_ids': ids, 'labels': labels, 'attention_mask': [1] * len(ids),
            'prompt_ids': prompt}


def collate(torch, examples, pad_id):
    require(examples and type(pad_id) is int, 'collation inputs')
    n = max(len(x['input_ids']) for x in examples)
    inputs = torch.full((len(examples), n), pad_id, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    for i, e in enumerate(examples):
        require(len(e['labels']) == len(e['input_ids']), 'label length mismatch')
        k = len(e['input_ids'])
        inputs[i, :k] = torch.tensor(e['input_ids'], dtype=torch.long)
        labels[i, :k] = torch.tensor(e['labels'], dtype=torch.long)
    # Deliberately preserves pinned upstream collator semantics, including EOS
    # attention positions when tokenizer fallback makes pad == eos.
    return {'input_ids': inputs, 'labels': labels, 'attention_mask': inputs.ne(pad_id)}


def sequence_nll(torch, logits, labels):
    import torch.nn.functional as F
    require(logits.ndim == 3 and tuple(logits.shape[:2]) == tuple(labels.shape), 'logit/label dimensions')
    y = labels[..., 1:].contiguous()
    require(bool(y.ne(-100).any(dim=-1).all()), 'sequence has no answer labels')
    require(bool(torch.isfinite(logits).all()), 'nonfinite logits')
    # Same dtype and sum reduction as pinned upstream compute_batch_nll.
    return F.cross_entropy(logits[..., :-1, :].contiguous().transpose(1, 2), y,
                           ignore_index=-100, reduction='none').sum(dim=-1)


def npo_loss(torch, student_logits, teacher_logits, labels, beta=0.1):
    import torch.nn.functional as F
    require(math.isfinite(beta) and beta > 0, 'invalid NPO beta')
    student = sequence_nll(torch, student_logits, labels)
    teacher = sequence_nll(torch, teacher_logits.detach(), labels)
    return (2.0 / beta) * F.softplus(-beta * (student - teacher)).mean()


def freeze_reference(reference):
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    return reference


def objective(torch, model, microbatch, reference=None, beta=0.1, alpha=1.0, gamma=1.0):
    if reference is None:
        loss = model(**microbatch['anchor']).loss
    else:
        require(model is not reference, 'student and teacher must be distinct objects')
        require(not reference.training and not any(p.requires_grad for p in reference.parameters()), 'teacher not frozen')
        student = model(**microbatch['anchor'])
        with torch.no_grad():
            teacher = reference(**microbatch['anchor'])
        forget = npo_loss(torch, student.logits, teacher.logits, microbatch['anchor']['labels'], beta)
        retain = model(**microbatch['retain']).loss
        loss = gamma * forget + alpha * retain
    require(loss.ndim == 0 and bool(torch.isfinite(loss)), 'nonfinite/scalar loss')
    return loss


def update(torch, model, optimizer, microbatches, reference=None, max_grad_norm=1.0, **kwargs):
    require(microbatches and 0 < max_grad_norm < float('inf'), 'invalid update inputs')
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for micro in microbatches:
        loss = objective(torch, model, micro, reference=reference, **kwargs)
        losses.append(float(loss.detach()))
        (loss / len(microbatches)).backward()
    params = list(model.parameters())
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params), 'nonfinite gradient')
    norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in params), 'nonfinite updated parameter')
    return {'loss_mean': sum(losses) / len(losses), 'microbatches': len(microbatches),
            'gradient_norm_before_clip': float(norm), 'normalization': 'actual_microbatch_count'}


def state_sha(torch, model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, list(t.shape), str(t.dtype)]).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_resume(torch, path, model, optimizer, cursor, contract_sha, schedule_sha):
    path = Path(path)
    require(not path.exists(), 'checkpoint exists')
    require(type(cursor) is int and cursor >= 0, 'invalid resume cursor')
    require(all(p.grad is None for p in model.parameters()), 'checkpoint must be at clean update boundary')
    require(all(p.device.type == 'cpu' for p in model.parameters()), 'CPU checkpoint helper is not a CUDA checkpoint implementation')
    payload = {'schema': 'm2_cpu_resume_v1', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'cursor': cursor, 'contract_sha': contract_sha, 'schedule_sha': schedule_sha,
               'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate()}
    with path.open('xb') as f:
        torch.save(payload, f)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_resume(torch, path, expected_sha, model, optimizer, contract_sha, schedule_sha):
    path = Path(path)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha, 'resume bytes changed')
    p = torch.load(path, map_location='cpu', weights_only=True)
    require(p['schema'] == 'm2_cpu_resume_v1' and p['contract_sha'] == contract_sha and
            p['schedule_sha'] == schedule_sha, 'resume identity mismatch')
    model.load_state_dict(p['model'], strict=True)
    optimizer.load_state_dict(p['optimizer'])
    torch.set_rng_state(p['torch_rng'])
    random.setstate(p['python_rng'])
    return p['cursor']
