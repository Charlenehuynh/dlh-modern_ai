#!/usr/bin/env python3
"""Generate a caption for an image using a pre-trained BLIP model."""

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def image_captioner(model, image_path, max_new_tokens):
    """
    Generates a textual description of an image with BLIP.
        caption (str): generated textual description of the image
    """
    processor = BlipProcessor.from_pretrained(model)
    blip_model = BlipForConditionalGeneration.from_pretrained(model)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    output_ids = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption
