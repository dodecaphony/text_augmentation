from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from data_preprocessing import data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


def tokenize_function(examples, m_type):
    return tokenizer(examples[m_type], padding="max_length", truncation=True)


tokenized_datasets = data.map(tokenize_function, batched=True)


train_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))
test_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000, 1200))

training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.evaluate()
