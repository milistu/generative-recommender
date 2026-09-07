import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration

from tiger.dataset import TigerDataset


def tokens_to_asin(
    tokens: list[int],
    sid_to_asin: dict[tuple[int, ...], str],
    codebook_size: int = 256,
    num_levels: int = 4,
) -> str | None:
    """Convert generated token IDs back to an ASIN via semantic ID lookup.

    The model generates offset tokens where level l uses range
    [l * codebook_size, (l + 1) * codebook_size - 1].

    Args:
        tokens: Generated token IDs (one per level, length == num_levels).
        sid_to_asin: Mapping from semantic ID tuple to ASIN.
        codebook_size: Number of codes per codebook level.
        num_levels: Number of levels in the semantic ID.

    Returns:
        ASIN string if the semantic ID is valid, None otherwise.
    """
    if len(tokens) != num_levels:
        return None

    codes = []
    for level, token in enumerate(tokens):
        expected_min = level * codebook_size
        expected_max = (level + 1) * codebook_size - 1
        if not (expected_min <= token <= expected_max):
            return None
        codes.append(token - expected_min)

    return sid_to_asin.get(tuple(codes))


def compute_ranking_metrics(
    predicted_items: list[list[str]],
    ground_truth_items: list[str],
    at_k: list[int] | None = None,
) -> dict[str, float]:
    """Compute Recall@K and NDCG@K for leave-one-out evaluation.

    For leave-one-out (one relevant item per user):
    - Recall@K = 1 if GT in top-K, else 0. Averaged over all users.
    - NDCG@K = 1/log2(pos+1) if GT at position pos (1-indexed) <= K,
      else 0. IDCG@K = 1/log2(2) = 1, so NDCG simplifies to DCG.

    Args:
        predicted_items: Ranked predicted ASINs per user.
        ground_truth_items: Ground truth ASIN per user (same length).
        at_k: K values to compute (default [5, 10]).

    Returns:
        {"recall@5": float, "ndcg@5": float, "recall@10": float, ...}
    """
    if at_k is None:
        at_k = [5, 10]

    results = {}

    for k in at_k:
        recalls = []
        ndcgs = []

        for gt, preds in zip(ground_truth_items, predicted_items):
            top_k = preds[:k]
            if gt in top_k:
                recalls.append(1.0)
                pos = top_k.index(gt) + 1
                ndcgs.append(1.0 / np.log2(pos + 1))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)

        results[f"recall@{k}"] = float(np.mean(recalls))
        results[f"ndcg@{k}"] = float(np.mean(ndcgs))

    return results


@torch.no_grad()
def beam_search_retrieval(
    model: T5ForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    beam_size: int = 20,
    num_return_sequences: int | None = None,
    max_new_tokens: int = 4,
) -> torch.Tensor:
    """Run beam search to generate top-K semantic ID token sequences.

    Args:
        model: T5ForConditionalGeneration model in eval mode.
        input_ids: Encoder input tokens (batch, seq_len).
        attention_mask: Attention mask (batch, seq_len).
        beam_size: Beam search width.
        num_return_sequences: Sequences to return per input.
            Defaults to beam_size. Must be <= beam_size.
        max_new_tokens: Number of tokens to generate (== num_levels).

    Returns:
        Generated token sequences (batch, num_return_sequences,
        max_new_tokens).
    """
    if num_return_sequences is None:
        num_return_sequences = beam_size

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=beam_size,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        early_stopping=False,
        output_scores=False,
        return_dict_in_generate=False,
    )

    batch_size = input_ids.shape[0]
    generated = outputs[:, -max_new_tokens:]
    return generated.view(batch_size, num_return_sequences, -1)


def decode_semantic_ids(
    generated_tokens: torch.Tensor,
    sid_to_asin: dict[tuple[int, ...], str],
    codebook_size: int = 256,
    num_levels: int = 4,
) -> list[list[str]]:
    """Convert generated token sequences to valid, deduplicated ASIN lists.

    Handles:
    - Tokens outside valid semantic range (pad/EOS/user tokens)
    - Semantic IDs not found in sid_to_asin (novel combinations)
    - Duplicate ASINs from different beam paths converging

    Args:
        generated_tokens: (batch, num_return_sequences, num_levels).
        sid_to_asin: Mapping from semantic ID tuple to ASIN.
        codebook_size: Size of each codebook level.
        num_levels: Number of levels in semantic ID.

    Returns:
        List per user: valid ASINs in beam order (deduplicated, first
        occurrence kept).
    """
    batch_predictions = []

    for batch_idx in range(generated_tokens.shape[0]):
        seen_asins: set[str] = set()
        valid_asins: list[str] = []

        for seq_idx in range(generated_tokens.shape[1]):
            tokens = generated_tokens[batch_idx, seq_idx].tolist()
            asin = tokens_to_asin(tokens, sid_to_asin, codebook_size, num_levels)
            if asin is not None and asin not in seen_asins:
                seen_asins.add(asin)
                valid_asins.append(asin)

        batch_predictions.append(valid_asins)

    return batch_predictions


@torch.no_grad()
def evaluate(
    model: T5ForConditionalGeneration,
    dataset: TigerDataset,
    sid_to_asin: dict[tuple[int, ...], str],
    device: torch.device,
    batch_size: int = 256,
    beam_size: int = 20,
    at_k: list[int] | None = None,
    max_new_tokens: int = 4,
    verbose: bool = True,
) -> dict[str, float]:
    """Run full evaluation pipeline on a dataset split.

    Data flow per batch:
    1. Load (input_ids, attention_mask, labels) from DataLoader
    2. Run beam search -> top-K token sequences
    3. Decode tokens -> ASINs
    4. Collect all user predictions
    Then compute macro-averaged Recall@K and NDCG@K.

    Args:
        model: T5ForConditionalGeneration model in eval mode.
        dataset: TigerDataset (typically val or test split).
        sid_to_asin: Mapping from semantic ID tuple to ASIN.
        device: Device for inference.
        batch_size: Dataloader batch size.
        beam_size: Beam search width.
        at_k: K values for metrics.
        max_new_tokens: Number of tokens to generate (== num_levels).
        verbose: Show progress bar.

    Returns:
        {"recall@5": float, "ndcg@5": float, ...}
    """
    if at_k is None:
        at_k = [5, 10]

    model.eval()
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_predictions: list[list[str]] = []
    all_ground_truths: list[str] = []

    iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        # Step 1: Beam search
        generated = beam_search_retrieval(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            beam_size=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=max_new_tokens,
        )

        # Step 2: Decode tokens -> ASINs per user
        batch_preds = decode_semantic_ids(
            generated.cpu(),
            sid_to_asin,
            dataset.codebook_size,
            dataset.num_levels,
        )
        all_predictions.extend(batch_preds)

        # Step 3: Decode ground truth labels -> ASINs
        for label_seq in labels:
            gt_tokens = label_seq.tolist()
            gt_asin = tokens_to_asin(
                gt_tokens, sid_to_asin, dataset.codebook_size, dataset.num_levels
            )
            all_ground_truths.append(gt_asin if gt_asin else "UNKNOWN")

    # Step 4: Compute metrics
    return compute_ranking_metrics(all_predictions, all_ground_truths, at_k)
