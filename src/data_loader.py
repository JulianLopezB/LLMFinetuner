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

# Example usage:
# For a dataset from Hugging Face:
# data_loader = DataLoader('glue', 'mrpc', split='train', from_huggingface=True)
# dataset = data_loader.get_dataset()
# For a local JSON dataset:
# local_data_loader = DataLoader('/path/to/local/dataset.json', 'json', from_huggingface=False)
# local_dataset = local_data_loader.get_dataset()
