"""KL anchor baseline for the sequential A->B study.

During stage B, each micro-step adds weight * KL(teacher || student) on one Task A
*reference* prompt (reference_A.jsonl, reserved for preservation and never used
for evaluation). The teacher is the frozen stage-A adapter: its next-token
distributions at the response positions are cached before B training starts.
The number of stage-B examples and optimizer updates is unchanged, so runs sit on
the same matched-budget tradeoff curve as replay. Each step costs one extra
forward/backward pass on the anchor prompt.
"""
import random

try:
    from prototype.sequential_replay import mode_of
except ModuleNotFoundError:
    from sequential_replay import mode_of


def anchor_examples(rows):
    """Turn reference rows into training-style {prompt, response} anchors."""
    anchors = []
    for row in rows:
        if row.get("group") != "protected_reference":
            raise ValueError(f"KL anchors must come from reference_A.jsonl, got {row.get('id')}")
        if mode_of(row["prompt"]) != "A":
            raise ValueError(f"KL anchor is not a Task A prompt: {row['prompt']}")
        anchors.append({"prompt": row["prompt"], "response": row["expected"]})
    if not anchors:
        raise ValueError("No KL anchor prompts supplied")
    return anchors


def anchor_schedule(n_anchors, n_steps, seed):
    """Anchor index for every micro-step: reshuffled passes over all anchors."""
    if n_anchors < 1 or n_steps < 0:
        raise ValueError("Need at least one anchor and a non-negative step count")
    rng = random.Random(seed + 7919)
    order = []
    while len(order) < n_steps:
        block = list(range(n_anchors))
        rng.shuffle(block)
        order.extend(block)
    return order[:n_steps]


def response_positions(labels):
    """Logit positions that predict supervised tokens (logits at t-1 predict token t)."""
    positions = [i - 1 for i, label in enumerate(labels) if label != -100 and i > 0]
    if not positions:
        raise ValueError("Anchor has no supervised response tokens")
    return positions


def teacher_log_probs(model, encoded, device, torch):
    """Cache the frozen stage-A model's log-probs at each anchor's response positions."""
    cached = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for ids, labels in encoded:
            positions = response_positions(labels)
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, positions]
            cached.append(torch.log_softmax(logits.float(), dim=-1).cpu())
    if was_training:
        model.train()
    return cached


def kl_term(model, ids, labels, teacher_logp, device, torch):
    """Mean over response positions of KL(teacher || student)."""
    positions = response_positions(labels)
    logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, positions]
    student_logp = torch.log_softmax(logits.float(), dim=-1)
    teacher = teacher_logp.to(device)
    return (teacher.exp() * (teacher - student_logp)).sum(-1).mean()


# Raw-activation anchor: keep hidden states at one layer close to the frozen
# stage-A model on the same reserved reference prompts. Representation-level
# counterpart to the output-level KL anchor; SAE-feature preservation would
# replace the raw hidden state with SAE codes.
def teacher_hidden(model, encoded, layer, device, torch):
    """Cache the frozen stage-A model's layer-`layer` hidden states at response positions."""
    cached = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for ids, labels in encoded:
            positions = response_positions(labels)
            out = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True)
            cached.append(out.hidden_states[layer][0, positions].float().cpu())
    if was_training:
        model.train()
    return cached


def act_term(model, ids, labels, teacher_h, layer, device, torch):
    """Relative squared drift ||h - h_A||^2 / ||h_A||^2 at the anchor's response positions."""
    positions = response_positions(labels)
    out = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True)
    student = out.hidden_states[layer][0, positions].float()
    teacher = teacher_h.to(device)
    return ((student - teacher) ** 2).sum() / (teacher ** 2).sum().clamp_min(1e-8)
