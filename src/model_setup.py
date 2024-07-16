from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch

class ModelSetup:
    def __init__(self, model_name, quantization_config=None, device_map=None, use_auth_token=False, peft_config=None):
        """
        Initialize the ModelSetup with the model's name and optional quantization and device mapping.

        Args:
            model_name (str): The name of the model to load (e.g., 'gpt2', 'bert-base-uncased').
            quantization_config (dict, optional): Configuration for model quantization.
            device_map (dict, optional): Mapping of model layers to specific devices.
            use_auth_token (bool, optional): Whether to use an authentication token for loading the model.
            peft_config (dict, optional): Configuration for PEFT.
        """
        self.model_name = model_name
        self.quantization_config = dict(quantization_config) if quantization_config else None
        self.device_map = device_map if device_map else {0: ""}
        self.use_auth_token = use_auth_token
        self.peft_config = peft_config
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
                load_in_4bit=self.quantization_config.get('load_in_4bit', True),
                bnb_4bit_use_double_quant=self.quantization_config.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=self.quantization_config.get('bnb_4bit_quant_type', "nf4"),
                bnb_4bit_compute_dtype=eval(f"torch.{self.quantization_config.get('bnb_4bit_compute_dtype', 'bfloat16')}")
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                quantization_config=bnb_config, 
                device_map=self.device_map,
                use_auth_token=self.use_auth_token
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                device_map=self.device_map,
                use_auth_token=self.use_auth_token
            )

        if self.peft_config:
            lora_config = LoraConfig(
                r=self.peft_config.get('r', 8),
                lora_alpha=self.peft_config.get('lora_alpha', 32),
                target_modules=self.peft_config.get('target_modules', ["q_proj", "v_proj"]),
                lora_dropout=self.peft_config.get('lora_dropout', 0.05),
                bias=self.peft_config.get('bias', "none"),
                task_type=self.peft_config.get('task_type', "CAUSAL_LM")
            )
            model = get_peft_model(model, lora_config)

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer corresponding to the model from Hugging Face's repository.

        Returns:
            PreTrainedTokenizer: The loaded tokenizer.
        """
        return AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.use_auth_token)

    def get_model_and_tokenizer(self):
        """
        Get both the model and tokenizer.

        Returns:
            tuple: A tuple containing both the loaded model and tokenizer.
        """
        return self.model, self.tokenizer

# Example usage:
# config = {
#     'model': {
#         'name': 'gpt2',
#         'quantization': {
#             'load_in_4bit': True,
#             'bnb_4bit_use_double_quant': True,
#             'bnb_4bit_quant_type': "nf4",
#             'bnb_4bit_compute_dtype': 'bfloat16'
#         },
#         'device_map': {'auto': True},
#         'peft_config': {
#             'r': 8,
#             'lora_alpha': 32,
#             'target_modules': ["linear"],
#             'lora_dropout': 0.05,
#             'bias': "none",
#             'task_type': "CAUSAL_LM"
#         }
#     }
# }
# setup = ModelSetup(
#     model_name=config['model']['name'],
#     quantization_config=config['model']['quantization'],
#     device_map=config['model']['device_map'],
#     use_auth_token=True,
#     peft_config=config['model']['peft_config']
# )
# model, tokenizer = setup.get_model_and_tokenizer()
