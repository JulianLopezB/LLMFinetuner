from transformers import PreTrainedModel
from huggingface_hub import HfApi, HfFolder
from peft import PeftModel  # Assuming peft module provides functionality to handle PEFT model operations
import os

class HuggingFaceIntegration:
    def __init__(self, model: PreTrainedModel, model_id: str, new_model: str, organization: str = None):
        """
        Initialize the integration setup with Hugging Face.

        Args:
            model (PreTrainedModel): The model to be pushed to the Hugging Face Hub.
            model_id (str): The repository name for the model on the Hub.
            organization (str, optional): The organization under which the model should be uploaded.
                                         If None, the model will be uploaded under the user's namespace.
        """
        self.model = model
        self.model_id = model_id
        self.new_model = new_model  # Save this as a separate attribute
        self.organization = organization
        self.api = HfApi()
        self.token = HfFolder.get_token()

    def save_and_merge_model(self, base_model: PreTrainedModel):
        """
        Save and merge PEFT model with a base model.

        Args:
            base_model (PreTrainedModel): The base model to merge the PEFT model with.
        """
        # Save the PEFT model
        peft_model = PeftModel.from_pretrained(self.model)
        peft_model.save_pretrained(self.model_id)

        # Merge and unload the model
        merged_model = peft_model.merge_and_unload(base_model)
        merged_model.save_pretrained("merged_model", safe_serialization=True)

    def push_to_hub(self):
        """
        Push the model to the Hugging Face Hub.

        Raises:
            ValueError: If the token is not found or invalid.
        """
        if not self.token:
            raise ValueError("Hugging Face token not found. Please login using `huggingface-cli login`.")

        repo_url = self.api.create_repo(
                name=self.new_model,
            token=self.token,
            organization=self.organization,
            private=False  # Set to True to create a private repository
        )

        self.model.push_to_hub(repo_url, use_auth_token=self.token)
        print(f"Model successfully uploaded to {repo_url}.")

        
# Example usage:
# model_setup = ModelSetup('bert-base-uncased')
# model, _ = model_setup.get_model_and_tokenizer()
# base_model = AutoModelForCausalLM.from_pretrained('bert-base-uncased', return_dict=True)
# hf_integration = HuggingFaceIntegration(model, 'my-awesome-peft-model')
# hf_integration.save_and_merge_model(base_model)
# hf_integration.push_to_hub()
