from allennlp_models.rc.tools import squad

def compute_metrics(pred, tokenizer):
    """Training metrics for sequence-to-sequence encoder-decoder.
    The tokenizer must be fixed, e.g. using ``functools.partial``."""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    return squad.compute_f1(pred_str, label_str)
