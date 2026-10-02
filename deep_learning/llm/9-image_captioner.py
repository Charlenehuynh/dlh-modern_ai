#!/usr/bin/env python3
"""Generate a caption for an image using a pre-trained BLIP model."""

import transformers
import PIL


def image_captioner(model, image_path, max_new_tokens):
    """
    Generates a textual description of an image with BLIP.

    Args:
        model (str): name of the pre-trained image captioning model
        image_path (str): path to the image file to caption
        max_new_tokens (int): maximum number of tokens to generate

    Returns:
        caption (str): generated textual description of the image
    """
    processor = transformers.BlipProcessor.from_pretrained(model)
    blip_model = transformers.BlipForConditionalGeneration.from_pretrained(model)

    image = PIL.Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    output_ids = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption
