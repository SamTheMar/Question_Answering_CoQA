def prepare_train_features(examples, tokenizer, max_length=380, doc_stride=128):
    """Tokenize our examples with truncation and padding, but keep the overflows using a
    stride.

    This results in one example possible giving several features when a context is long,
    each of those features having a context that overlaps a bit the context of the previous
    feature.

    To be used in the map method of ``datasets.DatasetDict``. Before using it, fix all the
    parameters using ``functools.partial``.

    Parameters
    ----------
    examples : datasets.DatasetDict
    tokenizer : transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        pretrained tokenizer for the model for which to use the features.
    max_length : int, optional
        maximum length of the tokenization, by default 380.
    doc_stride : int, optional
        The authorized overlap between two part of the context when splitting, by default 128.

    Returns
    -------
    datasets.DatasetDict
        tokenized examples.
    """
    question = [q.lstrip() for q in examples["question"]]
    story = [c.lstrip() for c in examples["story"]]

    tokenized_examples = tokenizer(
        question,
        story,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a
    # map from a feature to its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original
    # context. This will help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what
        # is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this
        # span of text.
        sample_index = sample_mapping[i]

        # Start/end character index of the answer in the text.
        start_char = examples["span_start"][sample_index]
        end_char = examples["span_end"][sample_index]

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the
        # CLS index).
        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the
            # answer.
            # Note: we could go after the last offset if the answer is the last word (edge
            # case).
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples
