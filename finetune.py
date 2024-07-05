from src.data_loader import DataLoader
from src.model_setup import ModelSetup
from src.trainer import CustomTrainer
from src.evaluator import Evaluator
from src.huggingface_integration import HuggingFaceIntegration
from src.config import Config
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./config", config_name="finetuning_config")
def main(config: DictConfig):
    # Load the dataset
    data_loader = DataLoader(config.dataset.path, from_huggingface=config.dataset.from_huggingface)
    train_dataset, eval_dataset = data_loader.get_dataset()['train'].train_test_split(test_size=0.2).values()

    # Setup the model and tokenizer with quantization and device configuration if required
    model_setup = ModelSetup(
        config.model.name,
        quantization_config=config.model.quantization,
        device_map=config.model.device_map
    )
    model, tokenizer = model_setup.get_model_and_tokenizer()

    # Configure PEFT if enabled
    if config.training.peft_enabled:
        lora_config = config.gtraining.lora_config
    else:
        lora_config = None

    # Setup and run the trainer
    trainer = CustomTrainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config.training.output_dir,
        peft_config=lora_config,
        **config.training.trainer_args
    )
    trainer.train()

    # Evaluate the model
    evaluator = Evaluator(model, tokenizer, eval_dataset)
    results = evaluator.evaluate()
    print("Evaluation Results:", results)

    # If Hugging Face push is enabled
    if config.training.hf_push:
        hf_integration = HuggingFaceIntegration(
            model,
            config.model.name,
            config.model.new_model,
            config.training.hf_org
        )
        hf_integration.save_and_push_model()


if __name__ == "__main__":
    main()
