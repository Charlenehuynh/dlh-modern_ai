#!/usr/bin/env python3
"""creates a high-level interface for performing text generation"""


def create_text_generator(
    model_name,
    prompt,
    max_new_tokens,
    temperature,
    repetition_penalty,
    no_repeat_ngram_size,
):
    """
    Returns:
    generator: A Hugging Face pipeline object.
    output (list[dict]): List of generated text predictions from the model.
    """
    generator = transformers.pipeline("text-generation", model=model_name)
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return generator, output
