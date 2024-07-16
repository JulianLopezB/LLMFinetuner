# LLMFinetuner

## Overview
This project is designed to fine-tune a pre-trained language model using the Hugging Face Transformers library. The model is configured for causal language modeling with additional support for quantization and PEFT (Parameter Efficient Fine-Tuning) using LoRA (Low-Rank Adaptation).

## Project Structure
- `config/`: Contains YAML configuration files for model and training setups.
- `src/`: Source code for model setup, data loading, training, evaluation, and integration with Hugging Face Hub.
- [setup_environment.py](file:///Users/julian/Work/Stefanini/LLMFinetuner/setup_environment.py#1%2C1-1%2C1): Script to install necessary packages and check environment setup.
- [finetune.py](file:///Users/julian/Work/Stefanini/LLMFinetuner/finetune.py#1%2C1-1%2C1): Main script to run the fine-tuning process.

## Configuration
The model and training configurations are specified in YAML format under the `config/` directory. For example, the configuration for fine-tuning can be found in:

```1:36:config/finetuning_config.yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.1"
  new_model: "Mistral-7B-Instruct-detcext-v0.1"  # Specify the new model name for saving or pushing
  quantization:
    load_in_4bit: true
    bnb_4bit_use_double_quant: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "bfloat16"
  device_map:
    0: ""  # Correct device map

dataset:
  path: "data/eval/example_instruction_dataset.jsonl"
  type: alpaca
  from_huggingface: false

training:
  output_dir: "./output"
  peft_enabled: true
  peft_config:
    r: 8
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]  # Update with correct target modules
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"
  hf_push: true
  hf_org: "my-organization"
  trainer_args:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 4
    max_steps: 100
    learning_rate: 0.0002
    logging_steps: 1
    save_strategy: "epoch"
    optim: "paged_adamw_8bit"
```


## Setup
Before running the fine-tuning script, ensure that all dependencies are installed and the environment is properly set up by running:
```python
python setup_environment.py
```
This script will handle the installation of required packages and check for necessary environment variables.

## Running the Fine-Tuning
To start the fine-tuning process, use the following command:
```bash
python finetune.py
```
This script initializes the model with the specified configuration, loads the dataset, and starts the training and evaluation process.

## Key Components
- **Data Loading**: Handles loading and splitting of datasets.
  
```1:42:src/data_loader.py
from datasets import load_dataset
from typing import Union
from pathlib import Path

class DataLoader:
    def __init__(self, dataset_name: Union[str, Path], file_format: str = 'json', from_huggingface: bool = True, **kwargs):
        """
        Initialize the DataLoader with dataset information.

        Args:
            dataset_name (Union[str, Path]): The name of the dataset or path to local dataset.
            file_format (str): The format of the dataset ('json', 'csv', etc.).
            from_huggingface (bool): Flag to determine if the dataset is loaded from Hugging Face.
            **kwargs: Additional keyword arguments for dataset configuration.
        """
        self.dataset_name = dataset_name
        self.file_format = file_format
        self.from_huggingface = from_huggingface
        self.dataset_kwargs = kwargs
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """
        Load a dataset based on initialization parameters.

        Returns:
            Dataset object as specified by the Hugging Face datasets library.
        """
        if self.from_huggingface:
            return load_dataset(self.dataset_name, **self.dataset_kwargs)
        else:
            return load_dataset(self.file_format, data_files=str(self.dataset_name), **self.dataset_kwargs)

    def get_dataset(self):
        """
        Retrieve the loaded dataset.

        Returns:
            Loaded dataset object.
        """
        return self.dataset
```

- **Model Setup**: Configures the model with optional quantization and device mapping.
  
```1:55:src/model_setup.py
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
import torch
```

- **Training**: Custom training logic including support for PEFT.
  
```1:64:src/trainer.py
from transformers import Trainer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import torch
```

- **Evaluation**: Evaluates the model on a specified dataset.
  
```1:29:src/evaluator.py
from transformers import Trainer, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

class Evaluator:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset: Dataset):
        """
        Initialize the Evaluator with the model, tokenizer, and evaluation dataset.

        Args:
            model (PreTrainedModel): The pre-trained model to be evaluated.
            tokenizer (PreTrainedTokenizer): The tokenizer for encoding the data.
            eval_dataset (Dataset): The dataset used for evaluation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.trainer = Trainer(model=self.model, tokenizer=self.tokenizer)

    def evaluate(self):
        """
        Evaluate the model using the provided evaluation dataset.

        Returns:
            dict: The evaluation results containing metrics like loss, accuracy, etc.
        """
        # This method can be expanded to handle custom metrics and specific evaluation logic
        evaluation_results = self.trainer.evaluate(eval_dataset=self.eval_dataset)
        return evaluation_results

```

- **Hugging Face Integration**: Manages saving and pushing models to the Hugging Face Hub.
  
```1:57:src/huggingface_integration.py
from transformers import PreTrainedModel
from huggingface_hub import HfApi, HfFolder
from peft import PeftModel  # Assuming peft module provides functionality to handle PEFT model operations
import os
```


## Output
The trained models and evaluation results are saved in the directory specified in the training configuration. Note that the output directory is ignored by git as specified in the `.gitignore` file:
```gitignore
startLine: 1
endLine: 1
```

## Additional Notes
Ensure that all necessary environment variables are set, such as `HUGGINGFACE_TOKEN`, to enable integration with Hugging Face services.

For any issues or further customization, refer to the individual scripts and configuration files within the project.
