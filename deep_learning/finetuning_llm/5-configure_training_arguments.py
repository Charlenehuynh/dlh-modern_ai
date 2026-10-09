#!/usr/bin/env python3
"""Module that configures Hugging Face TrainingArguments."""
import transformers


def configure_training_args(output_dir, epochs, per_device_train_batch_size,
                            per_device_eval_batch_size, learning_rate,
                            weight_decay, metric_for_best_model, seed):
    """
    Sets up the training arguments for a Hugging Face Trainer.

    Args:
        output_dir (str): Directory to save checkpoints and logs.
        epochs (int): Number of training epochs.
        per_device_train_batch_size (int): Training batch size per device.
        per_device_eval_batch_size (int): Evaluation batch size per device.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.
        metric_for_best_model (str): Metric used to pick the best checkpoint.
        seed (int): Random seed for reproducibility.

    Returns:
        transformers.TrainingArguments object.
    """
    return transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        push_to_hub=True,
        seed=seed
    )
