from transformers import Trainer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import torch

class CustomTrainer:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_dataset: Dataset, eval_dataset: Dataset, output_dir: str, peft_config=None, **training_args):
        """
        Initialize the CustomTrainer with the necessary components and PEFT configurations.

        Args:
            model (PreTrainedModel): The model to be trained.
            tokenizer (PreTrainedTokenizer): The tokenizer for encoding the data.
            train_dataset (Dataset): The dataset used for training.
            eval_dataset (Dataset): The dataset used for evaluation.
            output_dir (str): Directory where the model checkpoints will be saved.
            peft_config (LoraConfig, optional): Configuration for PEFT training.
            **training_args: Additional training arguments.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.training_args = TrainingArguments(output_dir=output_dir, **training_args)
        self.trainer = self.setup_trainer()

        if peft_config:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, peft_config)

    def setup_trainer(self):
        """
        Setup the Hugging Face Trainer with custom or PEFT configurations.

        Returns:
            Trainer: A configured Trainer instance.
        """
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                        'labels': torch.stack([f['input_ids'] for f in data])}
        )

    def train(self):
        """
        Start the training process.
        """
        self.trainer.train()

    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            dict: The evaluation results.
        """
        return self.trainer.evaluate()

# Example usage:
# peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=['linear'], lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
# trainer = CustomTrainer(model, tokenizer, train_dataset, eval_dataset, 'output_dir', peft_config=peft_config, per_device_train_batch_size=1, gradient_accumulation_steps=4, max_steps=100)
# trainer.train()
# eval_results = trainer.evaluate()
