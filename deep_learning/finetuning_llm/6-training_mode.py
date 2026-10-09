#!/usr/bin/env python3
"""Module that trains, evaluates and saves a Hugging Face model."""
import transformers


def train_model(model, training_args, train_dataset, eval_dataset,
                test_dataset, tokenizer, data_collator, compute_metrics,
                model_save_name):
    """
    Fine-tunes a model, evaluates it on the test set and saves it.

    Args:
        model (transformers.PreTrainedModel): Pre-trained model to fine-tune.
        training_args (transformers.TrainingArguments): Training arguments.
        train_dataset (datasets.Dataset): Tokenized training dataset.
        eval_dataset (datasets.Dataset): Tokenized validation dataset.
        test_dataset (datasets.Dataset): Tokenized test dataset.
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer.
        data_collator (callable): Data collator for dynamic padding.
        compute_metrics (callable): Function that computes metrics.
        model_save_name (str): Directory to save the model and tokenizer.

    Returns:
        trainer, train_results, test_results
    """
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    train_results = trainer.train()
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    trainer.save_model(model_save_name)
    tokenizer.save_pretrained(model_save_name)

    return trainer, train_results, test_results
