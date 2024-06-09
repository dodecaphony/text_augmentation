import enum
from datasets import load_dataset
import pandas as pd


class DatasetsAvailable(enum.Enum):
    MultidomainGold = "Multidomain gold dataset. For more see `ai-forever/spellcheck_benchmark`."
    RUSpellRU = "Social media texts and blogs. For more see `ai-forever/spellcheck_benchmark`."
    MedSpellchecker = "Medical anamnesis. For more see `ai-forever/spellcheck_benchmark`."
    GitHubTypoCorpusRu = "Github commits. For more see `ai-forever/spellcheck_benchmark`."


def load_available_dataset_from_hf(dataset_name: str, split: str = None) -> pd.DataFrame:
    if dataset_name not in [dataset.name for dataset in DatasetsAvailable]:
        raise ValueError(f"You provided wrong dataset name: {dataset_name}")
    dataset = load_dataset("ai-forever/spellcheck_benchmark", dataset_name, split=split)
    if split is None:
        dataset = pd.concat([dataset[split].to_pandas() for split in dataset.keys()]).reset_index(drop=True)
    else:
        dataset = dataset.to_pandas()
    return dataset


dataset_name = "GitHubTypoCorpusRu"
data = load_available_dataset_from_hf(dataset_name)

pd.set_option('display.max_colwidth', None)

print(data['source'].to_string(index=False, header=False))
# print(data['correction'].to_string(index=False, header=False)
