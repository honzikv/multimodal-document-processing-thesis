def prepare_examples(examples, vit_feature_extractor):
    images = examples['image']
    labels = examples['label']

    vit_encoding = vit_feature_extractor(images, return_tensors='pt')
    return {
        'pixel_values': vit_encoding['pixel_values'],
        'label': labels,
        'bbox_features': examples['bbox_features'],
    }
