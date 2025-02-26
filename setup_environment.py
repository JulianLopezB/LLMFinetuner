import subprocess
import sys
import os

def install_packages():
    """
    Install necessary Python packages using pip.
    """
    required_packages = [
        "transformers>=4.31.0",
        "datasets>=2.4.0",
        "torch>=1.12.0",
        "accelerate>=0.21.0",
        "huggingface_hub>=0.12.0",
        "scipy",  # Add specific version if needed
        "peft",  # This could require a specific installation method or version
        "bitsandbytes>=0.40.2",  # For optimizing model training
        "python-dotenv>=0.19.0",  # Add python-dotenv to the list
        "hydra-core>=1.1.0"  # Add hydra-core to the list
    ]

    for package in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_environment():
    """
    Check for required environment variables or other system settings.
    """
    if 'google.colab' in sys.modules:
        from huggingface_hub import notebook_login
        notebook_login()
        if not huggingface_token:
            raise EnvironmentError("HUGGINGFACE_TOKEN is not set in Google Colab. Please set this secret.")
        os.environ['HUGGINGFACE_TOKEN'] = huggingface_token
    else:
        from dotenv import load_dotenv  # Import here to ensure it's available after installation
        load_dotenv()  # Load environment variables from .env file

    # Example check for CUDA availability for PyTorch
    import torch
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation of CUDA and NVIDIA drivers.")
    
    # Example of checking an environment variable
    if 'HUGGINGFACE_TOKEN' not in os.environ:
        raise EnvironmentError("HUGGINGFACE_TOKEN is not set. Please set this environment variable.")

def main():
    print("Installing required packages...")
    install_packages()
    print("Checking environment settings...")
    check_environment()
    print("Setup completed successfully.")

if __name__ == "__main__":
    main()
