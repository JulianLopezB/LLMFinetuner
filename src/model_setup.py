from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
import torch

class ModelSetup:
    def __init__(self, model_name: str, quantization_config=None, device_map=None):
        """
        Initialize the ModelSetup with the model's name and optional quantization and device mapping.

        Args:
            model_name (str): The name of the model to load (e.g., 'gpt2', 'bert-base-uncased').
            quantization_config (dict, optional): Configuration for model quantization.
            device_map (dict, optional): Mapping of model layers to specific devices.
        """
        self.model_name = model_name
        self.quantization_config = dict(quantization_config) if quantization_config else None
        self.device_map = device_map
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self) -> PreTrainedModel:
        """
        Load the model from Hugging Face's model repository with optional quantization and device mapping.

        Returns:
            PreTrainedModel: The loaded model.
        """
        if self.quantization_config:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.quantization_config.get('load_in_4bit', False),
                bnb_4bit_use_double_quant=self.quantization_config.get('bnb_4bit_use_double_quant', False),
                bnb_4bit_quant_type=self.quantization_config.get('bnb_4bit_quant_type', "nf4"),
                bnb_4bit_compute_dtype=eval(f"torch.{self.quantization_config.get('bnb_4bit_compute_dtype', 'bfloat16')}")
            )
            return AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, device_map=self.device_map)
        else:
            return AutoModelForCausalLM.from_pretrained(self.model_name)

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer corresponding to the model from Hugging Face's repository.

        Returns:
            PreTrainedTokenizer: The loaded tokenizer.
        """
        return AutoTokenizer.from_pretrained(self.model_name)

    def get_model_and_tokenizer(self):
        """
        Get both the model and tokenizer.

        Returns:
            tuple: A tuple containing both the loaded model and tokenizer.
        """
        return self.model, self.tokenizer

# Example usage:
# config = {'load_in_4bit': True, 'bnb_4bit_use_double_quant': True, 'bnb_4bit_quant_type': "nf4", 'bnb_4bit_compute_dtype': torch.bfloat16}
# setup = ModelSetup('gpt2', quantization_config=config, device_map={"": 0})
# model, tokenizer = setup.get_model_and_tokenizer()
