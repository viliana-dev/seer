"""
SAE (Sparse Autoencoder) tools for mechanistic interpretability with Gemma Scope.

Usage:
    from sae_tools import load_sae, analyze_prompt, compare_prompts, search_features, steer_with_feature

    # Analyze which SAE features fire on a prompt
    results = analyze_prompt(model, tokenizer, "I want to survive at all costs", layer=20)
    for r in results:
        print(f"Feature {r['feature']}: {r['activation']:.2f} - {r['explanation']}")

    # Compare two prompts to find differentially active features
    diff = compare_prompts(model, tokenizer, "I must survive", "I accept my fate", layer=20)
    print("More active in A:", diff["more_in_a"])
    print("More active in B:", diff["more_in_b"])

    # Search Neuronpedia for features by description
    features = search_features("self-preservation")
    for f in features:
        print(f"Layer {f['layer']} Feature {f['feature']}: {f['description']}")

    # Steer generation using a specific SAE feature
    sae = load_sae(layer=20)
    with steer_with_feature(model, sae, feature_idx=12345, strength=5.0):
        output = model.generate(input_ids, max_new_tokens=100)
"""

_sae_cache = {}
_explanation_cache = {}


def load_sae(layer, width="16k", device="cuda"):
    """
    Load a Gemma Scope SAE for a given layer and width.

    Args:
        layer: Layer index (non-negative)
        width: SAE width string (default "16k")
        device: Device to load onto (default "cuda")

    Returns:
        sae_lens.SAE object, cached after first load
    """
    from sae_lens import SAE

    key = (layer, width)
    if key in _sae_cache:
        return _sae_cache[key]

    sae = SAE.from_pretrained(
        release="gemma-scope-9b-pt-res-canonical",
        sae_id=f"layer_{layer}/width_{width}/canonical",
        device=device,
    )[0]

    _sae_cache[key] = sae
    return sae


def analyze_prompt(model, tokenizer, text, layer, sae=None, width="16k", top_k=10, position=-1):
    """
    Analyze which SAE features activate on a prompt at a given layer/position.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: Input string, or list of chat messages for chat models
        layer: Layer index (negative counts from end: -1 = last layer)
        sae: Pre-loaded SAE object (will load one if None)
        width: SAE width string (default "16k"), used if sae is None
        top_k: Number of top features to return (default 10)
        position: Token position to analyze (default -1 = last token)

    Returns:
        List of dicts: [{"feature": int, "activation": float, "explanation": str}, ...]
    """
    import torch

    # Tokenize
    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt")
    elif isinstance(text, list):
        formatted = tokenizer.apply_chat_template(
            text, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt")
    else:
        raise TypeError(f"text must be str or list of messages, got {type(text)}")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # Resolve layer index
    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1

    if layer < 0:
        layer = num_layers + layer
    if not 0 <= layer < num_layers:
        raise IndexError(f"layer out of range [0, {num_layers})")

    hidden = hidden_states[layer + 1]
    seq_len = hidden.shape[1]

    # Resolve position
    pos = position
    if pos < 0:
        pos = seq_len + pos
    if not 0 <= pos < seq_len:
        raise IndexError(f"position out of range [0, {seq_len})")

    activation = hidden[0, pos, :].detach()  # shape (d_model,)

    # Load or reuse SAE
    if sae is None:
        sae = load_sae(layer, width, device=str(activation.device))

    # Encode through SAE
    feature_acts = sae.encode(activation.unsqueeze(0).unsqueeze(0))  # (1, 1, n_features)
    feature_acts = feature_acts.squeeze(0).squeeze(0)  # (n_features,)

    # Top-k features
    values, indices = torch.topk(feature_acts, k=min(top_k, feature_acts.shape[0]))

    results = []
    for idx, val in zip(indices.tolist(), values.tolist()):
        explanation = get_feature_explanation(idx, layer, width)
        results.append({
            "feature": idx,
            "activation": round(val, 4),
            "explanation": explanation,
        })

    return results


def get_feature_explanation(feature_idx, layer, width="16k", model_id="gemma-2-9b-it"):
    """
    Fetch the Neuronpedia explanation for a specific SAE feature.

    Args:
        feature_idx: Feature index in the SAE
        layer: Layer number
        width: SAE width string (default "16k")
        model_id: Neuronpedia model identifier (default "gemma-2-9b-it")

    Returns:
        Explanation string, or "(no explanation available)" on error
    """
    import requests

    cache_key = (model_id, layer, width, feature_idx)
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    try:
        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}-gemmascope-res-{width}/{feature_idx}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        explanations = data.get("explanations", [])
        if explanations:
            explanation = explanations[0].get("description", "(no explanation available)")
        else:
            explanation = "(no explanation available)"
    except Exception:
        explanation = "(no explanation available)"

    _explanation_cache[cache_key] = explanation
    return explanation


def search_features(query, model_id="gemma-2-9b-it"):
    """
    Search Neuronpedia for SAE features matching a text query.

    Args:
        query: Search string (e.g. "self-preservation", "deception")
        model_id: Neuronpedia model identifier (default "gemma-2-9b-it")

    Returns:
        List of dicts: [{"feature": int, "layer": str, "description": str, "index": str}, ...]
    """
    import requests

    try:
        url = "https://www.neuronpedia.org/api/explanation/search-model"
        resp = requests.post(url, json={"modelId": model_id, "query": query}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "feature": item.get("index"),
                "layer": item.get("layer"),
                "description": item.get("description", ""),
                "index": item.get("index"),
            })
        return results
    except Exception:
        return []


def compare_prompts(model, tokenizer, text_a, text_b, layer, sae=None, width="16k", top_k=10, position=-1):
    """
    Compare SAE feature activations between two prompts.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text_a: First prompt (string or chat messages)
        text_b: Second prompt (string or chat messages)
        layer: Layer index (negative counts from end)
        sae: Pre-loaded SAE object (will load one if None)
        width: SAE width string (default "16k"), used if sae is None
        top_k: Number of top differential features per direction (default 10)
        position: Token position to analyze (default -1 = last token)

    Returns:
        Dict with "more_in_a" and "more_in_b" lists, each containing
        [{"feature": int, "diff": float, "activation_a": float, "activation_b": float, "explanation": str}, ...]
    """
    import torch

    def _encode(text):
        if isinstance(text, str):
            inputs = tokenizer(text, return_tensors="pt")
        elif isinstance(text, list):
            formatted = tokenizer.apply_chat_template(
                text, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt")
        else:
            raise TypeError(f"text must be str or list of messages, got {type(text)}")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states) - 1
        resolved_layer = layer
        if resolved_layer < 0:
            resolved_layer = num_layers + resolved_layer
        if not 0 <= resolved_layer < num_layers:
            raise IndexError(f"layer out of range [0, {num_layers})")

        hidden = hidden_states[resolved_layer + 1]
        seq_len = hidden.shape[1]
        pos = position
        if pos < 0:
            pos = seq_len + pos
        if not 0 <= pos < seq_len:
            raise IndexError(f"position out of range [0, {seq_len})")

        return hidden[0, pos, :].detach(), resolved_layer

    act_a, resolved_layer = _encode(text_a)
    act_b, _ = _encode(text_b)

    if sae is None:
        sae = load_sae(resolved_layer, width, device=str(act_a.device))

    features_a = sae.encode(act_a.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    features_b = sae.encode(act_b.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    diff = features_a - features_b

    # Features more active in A
    vals_a, idxs_a = torch.topk(diff, k=min(top_k, diff.shape[0]))
    more_in_a = []
    for idx, val in zip(idxs_a.tolist(), vals_a.tolist()):
        explanation = get_feature_explanation(idx, resolved_layer, width)
        more_in_a.append({
            "feature": idx,
            "diff": round(val, 4),
            "activation_a": round(features_a[idx].item(), 4),
            "activation_b": round(features_b[idx].item(), 4),
            "explanation": explanation,
        })

    # Features more active in B
    vals_b, idxs_b = torch.topk(-diff, k=min(top_k, diff.shape[0]))
    more_in_b = []
    for idx, val in zip(idxs_b.tolist(), vals_b.tolist()):
        explanation = get_feature_explanation(idx, resolved_layer, width)
        more_in_b.append({
            "feature": idx,
            "diff": round(val, 4),
            "activation_a": round(features_a[idx].item(), 4),
            "activation_b": round(features_b[idx].item(), 4),
            "explanation": explanation,
        })

    return {"more_in_a": more_in_a, "more_in_b": more_in_b}


def steer_with_feature(model, sae, feature_idx, strength=1.0, layer=None):
    """
    Context manager that steers model generation using an SAE feature direction.

    Args:
        model: HuggingFace model
        sae: SAE object (from load_sae)
        feature_idx: Feature index to steer with
        strength: Multiplier for the steering vector (default 1.0)
        layer: Layer to hook (defaults to sae.cfg.hook_layer)

    Example:
        sae = load_sae(layer=20)
        with steer_with_feature(model, sae, feature_idx=12345, strength=5.0):
            output = model.generate(input_ids, max_new_tokens=100)
    """
    from contextlib import contextmanager

    @contextmanager
    def _hook():
        # Get feature direction from decoder weights
        direction = sae.W_dec[feature_idx].detach()

        # Determine target layer
        target_layer = layer if layer is not None else sae.cfg.hook_layer

        # Find decoder layers
        layers_module = None
        for attr_path in ["model.layers", "model.model.layers", "model.model.model.layers"]:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                layers_module = obj
                break
            except AttributeError:
                continue

        if layers_module is None:
            raise RuntimeError("Could not find decoder layers on model")

        if not 0 <= target_layer < len(layers_module):
            raise IndexError(f"layer {target_layer} out of range [0, {len(layers_module)})")

        def hook_fn(_, __, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h += strength * direction.to(h.device, h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h

        handle = layers_module[target_layer].register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    return _hook()
