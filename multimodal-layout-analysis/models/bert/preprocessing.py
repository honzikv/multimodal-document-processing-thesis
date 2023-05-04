def prepare_examples(examples, tokenizer):
    """
    Prepare examples for the network
    """

    words = [' '.join(sequence) for sequence in examples['tokens']]
    # Replace [MASK] with tokenizer.mask_token
    words = [word.replace('[MASK]', tokenizer.mask_token) for word in words]
    labels = examples['label']

    encoding = tokenizer(words, padding='max_length', truncation=True, return_tensors='pt')
    encoding['label'] = labels

    return encoding
