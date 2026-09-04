"""Loss functions over the tape, selected by `loss_function` in the training config.

All of them split the tokens into two groups by the dataset's mask: response text (mask 1) and
state transition tokens (mask >= 2), so the two can be weighted independently per source.
"""

import torch
import torch.nn.functional as F
from functools import partial
from utils import compute_logit_adjustments_tensor, compute_all_logit_adjustments_tensor


def compute_balanced_loss(batch, model_outputs, weight_lookup, normalize_by_count):
    """
    Computes loss with flexible weighting strategies.

    Args:
        batch (dict): Batch data containing 'input_ids', 'mask', 'dataset_id'.
        model_outputs: Model output containing logits.
        weight_lookup (Tensor): [Num_Datasets, 2] tensor.
        normalize_by_count (bool):
            - True: "Batch-Level Balance Mode".
              Response: pool all response tokens across the batch, compute weighted mean.
              Transition: compute weighted mean loss per mask_id class separately,
              then average across classes to handle inter-class token count imbalance.
            - False: "Token-Level Standard Mode".
              Apply weights to every token individually and compute global weighted mean.
    """
    labels = batch["input_ids"]  # In Causal LM, labels are input_ids
    mask = batch["mask"]
    logits = model_outputs.logits
    dataset_ids = batch["dataset_id"]

    # Shift inputs for Causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # 2. Prepare Weights [B, 2] -> [B, L_shift]
    # batch_weights[b] = [resp_weight, trans_weight]
    batch_weights = weight_lookup[dataset_ids.to(weight_lookup.device)]  # (B, 2)
    batch_size, seq_len = shift_labels.shape

    # 3. Flatten & Compute Raw Loss [B, L]
    # Calculate once, reuse everywhere
    raw_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").view(batch_size, seq_len)

    # 4. Define Masks [B, L]
    resp_mask = (shift_mask == 1).float()
    trans_mask = (shift_mask >= 2).float()

    # Strategy A: Batch-Level Balance (Normalize by Count, Per-Class Averaging)
    # For response tokens: pool all resp tokens, compute weighted mean.
    # For transition tokens: compute weighted mean per mask_id class separately,
    #   then average across classes (to handle class imbalance).
    if normalize_by_count:
        # 1. Response loss: weighted average over all resp tokens in the batch
        #    avg = Σ(loss_i * w_i) / Σ(w_i)
        resp_w_map = batch_weights[:, 0].unsqueeze(1) * resp_mask  # [B, L]
        total_weighted_loss_r = (raw_losses * resp_w_map).sum()
        resp_w_sum = resp_w_map.sum().clamp(min=1e-9)
        avg_loss_r = total_weighted_loss_r / resp_w_sum

        # 2. Transition loss: per-class weighted average, then average across classes
        trans_w = batch_weights[:, 1].unsqueeze(1)  # [B, 1]
        unique_mask_ids = shift_mask[shift_mask >= 2].unique()
        num_trans_classes = 0
        sum_class_means = torch.zeros(1, device=raw_losses.device)

        for mid in unique_mask_ids:
            class_mask = (shift_mask == mid).float()  # [B, L]
            class_w_map = class_mask * trans_w  # [B, L]
            class_w_sum = class_w_map.sum()
            if class_w_sum > 0:
                class_weighted_loss = (raw_losses * class_w_map).sum()
                sum_class_means = sum_class_means + class_weighted_loss / class_w_sum
                num_trans_classes += 1

        avg_loss_t = sum_class_means / max(num_trans_classes, 1)

        return avg_loss_r + avg_loss_t, {"response_loss": avg_loss_r.item(), "transition_loss": avg_loss_t.item()}

    # Strategy B: Token-Level Standard (Global Weighted Mean)
    # Target: Final = Sum(L_ij * W_ij) / Sum(W_ij)
    else:
        # Expand weights to [B, L]
        resp_w_map = batch_weights[:, 0].unsqueeze(1) * resp_mask
        trans_w_map = batch_weights[:, 1].unsqueeze(1) * trans_mask

        token_weights = resp_w_map + trans_w_map
        active_weight_sum = token_weights.sum().clamp(min=1e-9)

        # Calculate Weighted Sums
        total_weighted_loss_r = (raw_losses * resp_w_map).sum()
        total_weighted_loss_t = (raw_losses * trans_w_map).sum()

        # Normalize
        avg_loss_r = total_weighted_loss_r / active_weight_sum
        avg_loss_t = total_weighted_loss_t / active_weight_sum

        return avg_loss_r + avg_loss_t, {"response_loss": avg_loss_r.item(), "transition_loss": avg_loss_t.item()}


def compute_logit_adjusted_loss(batch, model_outputs, weight_lookup, logit_adjustments):
    """
    Computes a hybrid loss: standard Cross Entropy for 'response' tokens and Logit-Adjusted
    Cross Entropy for 'transition' tokens to correct for class imbalance.

    Mechanism:
    1. Response Tokens (mask == 1): Standard Cross Entropy loss.
    2. Transition Tokens (mask >= 2): Logit Adjusted Loss (logits + tau * log(prior)).
    3. Aggregation: Combines both components using dataset-specific weights defined in `weight_lookup`.

    Args:
        batch (dict): Contains 'input_ids', 'mask', and 'dataset_id'.
        model_outputs: The model output containing logits.
        weight_lookup (Tensor): [Num_Datasets, 2] mapping dataset IDs to [response_weight, transition_weight].
        logit_adjustments (Tensor): [Vocab_Size] tensor containing adjustment terms (tau * log(prior))
                                    for transition tokens.

    Returns:
        tuple: (final_loss, other_info_dict)
    """
    labels = batch["input_ids"]
    mask = batch["mask"]
    dataset_ids = batch["dataset_id"]
    logits = model_outputs.logits

    # 1. Shift inputs for Causal LM (Standard procedure)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # [B, L_shift]
    batch_size, seq_len = shift_labels.shape

    # 2. Build the weight matrix [B, L_shift]
    # 2.1 look up the per-sequence weight config for the current batch: [B, 2]
    # batch_weights[b] = [resp_weight, trans_weight]
    batch_weights = weight_lookup[dataset_ids.to(weight_lookup.device)]
    # 2.2 expand to the Token level [B, L_shift]
    token_weights = torch.zeros_like(shift_mask, dtype=torch.float32)
    # build the Response weights (Mask == 1)
    # batch_weights[:, 0] is [B]; unsqueeze to [B, 1], then broadcast via expand
    resp_w_expanded = batch_weights[:, 0].unsqueeze(1).expand(batch_size, seq_len)
    token_weights = torch.where(shift_mask == 1, resp_w_expanded, token_weights)
    # build the Transition weights (Mask >= 2)
    trans_w_expanded = batch_weights[:, 1].unsqueeze(1).expand(batch_size, seq_len)
    token_weights = torch.where(shift_mask >= 2, trans_w_expanded, token_weights)

    # 2. Flatten to [N, Vocab] and [N]
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)
    token_weights = token_weights.view(-1)

    # 3. Separate the batch into two groups using masks
    # Group A: Response tokens (Standard Loss)
    response_bool = shift_mask == 1
    # Group B: Transition tokens (Logit Adjusted Loss)
    transition_bool = shift_mask >= 2

    total_loss_sum = 0.0
    active_weight_sum = token_weights.sum() + 1e-9
    other_info = {"response_loss": 0.0, "transition_loss": 0.0}

    # 4. Calculate Loss for Response Tokens (Standard)
    if response_bool.any():
        # Select standard logits
        resp_logits = shift_logits[response_bool]
        resp_labels = shift_labels[response_bool]

        resp_loss_vec = F.cross_entropy(resp_logits, resp_labels, reduction="none")
        weighted_resp_loss = (resp_loss_vec * token_weights[response_bool]).sum()
        total_loss_sum += weighted_resp_loss

        other_info["response_loss"] = weighted_resp_loss.item()

    # 5. Calculate Loss for Transition Tokens (Logit Adjusted)
    # Based on Eq 10: logit = f(x) + tau * log(pi)
    if transition_bool.any():
        # Select logits and apply adjustment
        trans_logits = shift_logits[transition_bool]
        trans_labels = shift_labels[transition_bool]

        # Ensure adjustments are on the correct device and broadcast
        # trans_logits: [N_trans, V], adjustments: [V] -> broadcasting works
        adjusted_trans_logits = trans_logits + logit_adjustments.to(trans_logits.device)

        # Logit Adjusted Cross Entropy
        trans_loss_vec = F.cross_entropy(adjusted_trans_logits, trans_labels, reduction="none")

        weighted_trans_loss = (trans_loss_vec * token_weights[transition_bool]).sum()
        total_loss_sum += weighted_trans_loss

        other_info["transition_loss"] = weighted_trans_loss.item()

    # 6. Final Reduction (Mean over all valid tokens)
    if total_loss_sum > 0:
        for key in other_info:
            other_info[key] /= active_weight_sum
        return total_loss_sum / active_weight_sum, other_info
    else:
        dummy_loss = logits.sum() * 0.0
        return dummy_loss, other_info


def compute_all_logit_adjusted_loss(batch, model_outputs, weight_lookup, logit_adjustments):
    """
    Applies Logit-Adjusted Cross Entropy to BOTH response and transition tokens.
    - Response tokens: adjusted with a unified response class probability.
    - Transition tokens: adjusted with per-token probabilities.
    The logit_adjustments tensor should contain adjustments for ALL vocab positions
    (generated by compute_all_logit_adjustments_tensor).

    Args:
        batch (dict): Contains 'input_ids', 'mask', and 'dataset_id'.
        model_outputs: The model output containing logits.
        weight_lookup (Tensor): [Num_Datasets, 2] mapping dataset IDs to [response_weight, transition_weight].
        logit_adjustments (Tensor): [Vocab_Size] tensor containing adjustment terms for ALL vocab positions.

    Returns:
        tuple: (final_loss, other_info_dict)
    """
    labels = batch["input_ids"]
    mask = batch["mask"]
    dataset_ids = batch["dataset_id"]
    logits = model_outputs.logits

    # 1. Shift inputs for Causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    batch_size, seq_len = shift_labels.shape

    # 2. Build the weight matrix [B, L_shift]
    batch_weights = weight_lookup[dataset_ids.to(weight_lookup.device)]
    token_weights = torch.zeros_like(shift_mask, dtype=torch.float32)
    resp_w_expanded = batch_weights[:, 0].unsqueeze(1).expand(batch_size, seq_len)
    token_weights = torch.where(shift_mask == 1, resp_w_expanded, token_weights)
    trans_w_expanded = batch_weights[:, 1].unsqueeze(1).expand(batch_size, seq_len)
    token_weights = torch.where(shift_mask >= 2, trans_w_expanded, token_weights)

    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)
    token_weights = token_weights.view(-1)

    response_bool = shift_mask == 1
    transition_bool = shift_mask >= 2

    total_loss_sum = 0.0
    other_info = {"response_loss": 0.0, "transition_loss": 0.0}

    adj = logit_adjustments.to(shift_logits.device)

    # 3. Response Tokens (Logit Adjusted with unified response prob)
    if response_bool.any():
        resp_logits = shift_logits[response_bool] + adj
        resp_labels = shift_labels[response_bool]
        resp_loss_vec = F.cross_entropy(resp_logits, resp_labels, reduction="none")
        weighted_resp_loss = (resp_loss_vec * token_weights[response_bool]).mean()
        total_loss_sum += weighted_resp_loss
        other_info["response_loss"] = weighted_resp_loss.item()

    # 4. Transition Tokens (Logit Adjusted with per-token prob)
    if transition_bool.any():
        trans_logits = shift_logits[transition_bool] + adj
        trans_labels = shift_labels[transition_bool]
        trans_loss_vec = F.cross_entropy(trans_logits, trans_labels, reduction="none")
        weighted_trans_loss = (trans_loss_vec * token_weights[transition_bool]).mean()
        total_loss_sum += weighted_trans_loss
        other_info["transition_loss"] = weighted_trans_loss.item()

    if total_loss_sum > 0:
        return total_loss_sum, other_info
    else:
        dummy_loss = logits.sum() * 0.0
        return dummy_loss, other_info


def compute_fuzzy_logit_adjusted_loss(batch, model_outputs, weight_lookup, logit_adjustments, maskid_to_tokenid_map, response_exact_prob=0.0):
    """
    Computes loss with Logit Adjustment for transition tokens, and a sampled "Fuzzy vs Exact" loss for response tokens.
    """
    labels = batch["input_ids"]
    mask = batch["mask"]
    dataset_ids = batch["dataset_id"]
    logits = model_outputs.logits

    batch_size, _, vocab_size = logits.shape

    transition_vocab_mask = torch.zeros(vocab_size, dtype=torch.bool).to(logits.device)
    for transition_mask_id, token_id in maskid_to_tokenid_map.items():
        transition_vocab_mask[token_id] = True

    # 1. Shift inputs for Causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # 2. Build the weight matrix
    batch_size, seq_len = shift_labels.shape
    batch_weights = weight_lookup[dataset_ids.to(weight_lookup.device)]
    token_weights = torch.zeros_like(shift_mask, dtype=torch.float32)

    resp_w_expanded = batch_weights[:, 0].unsqueeze(1).expand(batch_size, seq_len)
    token_weights = torch.where(shift_mask == 1, resp_w_expanded, token_weights)

    trans_w_expanded = batch_weights[:, 1].unsqueeze(1).expand(batch_size, seq_len)
    token_weights = torch.where(shift_mask >= 2, trans_w_expanded, token_weights)

    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)
    token_weights = token_weights.view(-1)

    response_bool = shift_mask == 1
    transition_bool = shift_mask >= 2

    total_loss_sum = 0.0
    active_weight_sum = token_weights.sum() + 1e-9
    other_info = {"response_loss": 0.0, "transition_loss": 0.0, "response_exact_loss": 0.0, "response_fuzzy_loss": 0.0}

    # 3. Response Tokens Loss (Sampled)
    if response_bool.any():
        resp_logits = shift_logits[response_bool]
        resp_labels = shift_labels[response_bool]
        resp_weights = token_weights[response_bool]

        # generate the sampling Mask
        rand_tensor = torch.rand(resp_labels.shape, device=resp_labels.device)
        exact_mask = rand_tensor < response_exact_prob
        fuzzy_mask = ~exact_mask

        resp_loss_vec = torch.zeros_like(resp_labels, dtype=torch.float32)

        # 3.1 Exact Match Loss (Standard CE)
        if exact_mask.any():
            exact_logits = resp_logits[exact_mask]
            exact_labels = resp_labels[exact_mask]
            exact_loss = F.cross_entropy(exact_logits, exact_labels, reduction="none")
            resp_loss_vec[exact_mask] = exact_loss.to(torch.float32)
            other_info["response_exact_loss"] = (exact_loss * resp_weights[exact_mask]).sum().item()

        # 3.2 Fuzzy Match Loss (Predict ANY non-transition token)
        if fuzzy_mask.any():
            fuzzy_logits = resp_logits[fuzzy_mask]

            # logsumexp(all)
            lse_all = torch.logsumexp(fuzzy_logits, dim=-1)

            # logsumexp(response_only) by masking transitions to -inf
            fuzzy_logits_response_only = fuzzy_logits.clone()
            # transition_vocab_mask is a boolean tensor of shape [V]
            fuzzy_logits_response_only[:, transition_vocab_mask] = -float("inf")
            lse_response = torch.logsumexp(fuzzy_logits_response_only, dim=-1)

            # Loss = logsumexp(all) - logsumexp(response_only)
            fuzzy_loss = lse_all - lse_response
            resp_loss_vec[fuzzy_mask] = fuzzy_loss.to(torch.float32)
            other_info["response_fuzzy_loss"] = (fuzzy_loss * resp_weights[fuzzy_mask]).sum().item()

        weighted_resp_loss = (resp_loss_vec * resp_weights).sum()
        total_loss_sum += weighted_resp_loss
        other_info["response_loss"] = weighted_resp_loss.item()

    # 4. Transition Tokens Loss (Logit Adjusted)
    if transition_bool.any():
        trans_logits = shift_logits[transition_bool]
        trans_labels = shift_labels[transition_bool]

        adjusted_trans_logits = trans_logits + logit_adjustments.to(trans_logits.device)
        trans_loss_vec = F.cross_entropy(adjusted_trans_logits, trans_labels, reduction="none")

        weighted_trans_loss = (trans_loss_vec * token_weights[transition_bool]).sum()
        total_loss_sum += weighted_trans_loss
        other_info["transition_loss"] = weighted_trans_loss.item()

    if total_loss_sum > 0:
        for key in other_info:
            other_info[key] /= active_weight_sum.item()
        return total_loss_sum / active_weight_sum, other_info
    else:
        dummy_loss = logits.sum() * 0.0
        return dummy_loss, other_info


def compute_accuracy_counts(logits, labels, mask, maskid_to_tokenid_map):
    """
    Return correct_count and total_count to support distributed accumulation.
    """
    # Shift inputs
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # Get predictions
    predictions = shift_logits.argmax(dim=-1).view(-1)
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)

    correct_bool = predictions == shift_labels
    valid_mask = shift_mask != 0  # exclude the 0 mask from System Prompt and Padding

    results = {}

    for transition_mask_id in maskid_to_tokenid_map:
        mask_bool = shift_mask == transition_mask_id

        num_total = mask_bool.sum().item()
        num_correct = correct_bool[mask_bool].sum().item()

        tp = num_correct
        fn = num_total - tp
        fp = 0

        token_id = maskid_to_tokenid_map[transition_mask_id]
        fp_bool = (predictions == token_id) & (shift_mask != transition_mask_id) & valid_mask
        fp = fp_bool.sum().item()

        # only return tp, fp, fn
        results[transition_mask_id] = {"tp": tp, "fp": fp, "fn": fn}

    return results


def compute_accuracy_counts_per_sample(logits, labels, mask, maskid_to_tokenid_map):
    """
    Compute TP/FP/FN for each transition state per sample, for significance testing.
    Returns a list[dict] with one element per sequence in the batch.
    """
    # Shift inputs
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    batch_size = shift_logits.shape[0]
    predictions = shift_logits.argmax(dim=-1)  # (B, T)

    correct_bool = predictions == shift_labels  # (B, T)
    valid_mask = shift_mask != 0  # (B, T)

    per_sample_results = []
    for b in range(batch_size):
        sample_result = {}
        for transition_mask_id, token_id in maskid_to_tokenid_map.items():
            mask_bool = shift_mask[b] == transition_mask_id
            tp = correct_bool[b][mask_bool].sum().item()
            fn = mask_bool.sum().item() - tp
            fp_bool = (predictions[b] == token_id) & (shift_mask[b] != transition_mask_id) & valid_mask[b]
            fp = fp_bool.sum().item()
            sample_result[transition_mask_id] = {"tp": tp, "fp": fp, "fn": fn}
        per_sample_results.append(sample_result)

    return per_sample_results


def build_compute_loss_fn(
    train_args,
    accelerator,
    weight_lookup_tensor,
    maskid_to_tokenid_map,
    model_vocab_size=None,
    train_dataset=None,
):
    """Bind the loss named by `train_args["loss_function"]` to its configuration.

    Returns a ``compute_loss(batch, model_outputs)`` callable.
    """
    loss_type = train_args["loss_function"]
    if loss_type == "loss_modification":
        norm_by_count = train_args["normalize_by_count"]
        accelerator.print(f"Loss Config: normalize_by_count={norm_by_count}")
        compute_loss = partial(
            compute_balanced_loss,
            weight_lookup=weight_lookup_tensor,
            normalize_by_count=norm_by_count,  # pass in as a bool
        )

    elif loss_type == "logit_adjustment":
        logit_adjustments = compute_logit_adjustments_tensor(
            train_dataset,
            model_vocab_size=model_vocab_size,
            prob_denominator=train_args.get("prob_denominator", "all"),
            tau=train_args["logit_adjustment_tau"],
        )
        accelerator.print(f"Logit Adjustment tensor created. Shape: {logit_adjustments.shape}")
        compute_loss = partial(compute_logit_adjusted_loss, weight_lookup=weight_lookup_tensor, logit_adjustments=logit_adjustments)

    elif loss_type == "all_logit_adjustment":
        assert train_args["prob_denominator"] == "all"
        logit_adjustments = compute_all_logit_adjustments_tensor(
            train_dataset,
            model_vocab_size=model_vocab_size,
            prob_denominator="all",
            tau=train_args["logit_adjustment_tau"],
        )
        accelerator.print(f"All Logit Adjustment tensor created. Shape: {logit_adjustments.shape}")
        compute_loss = partial(compute_all_logit_adjusted_loss, weight_lookup=weight_lookup_tensor, logit_adjustments=logit_adjustments)

    elif loss_type == "fuzzy_logit_adjustment":
        logit_adjustments = compute_logit_adjustments_tensor(
            train_dataset,
            model_vocab_size=model_vocab_size,
            prob_denominator=train_args.get("prob_denominator", "all"),
            tau=train_args["logit_adjustment_tau"],
        )
        response_exact_prob = train_args["response_exact_prob"]
        accelerator.print(f"Loss Config: Fuzzy Logit Adjustment (p_exact={response_exact_prob})")

        compute_loss = partial(
            compute_fuzzy_logit_adjusted_loss,
            weight_lookup=weight_lookup_tensor,
            logit_adjustments=logit_adjustments,
            maskid_to_tokenid_map=maskid_to_tokenid_map,
            response_exact_prob=response_exact_prob,
        )

    else:
        raise ValueError(f"Unknown loss_function: {train_args['loss_function']}")

    return compute_loss
