import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import argparse
import evaluate
import os


def load_and_preprocess_data(jsonl_file):
    """Load and preprocess data from JSONL file with new format"""
    # Load the dataset from JSONL file
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
    
    # Filter for required fields and valid labels
    filtered_data = []
    for item in data:
        if all(key in item for key in ['title', 'abstract', 'isBionlp']) and item['isBionlp'] in ['Y', 'N']:
            filtered_data.append(item)
    
    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)
    
    # Combine title and abstract as input text
    df['text'] = 'Title: ' + df['title'].fillna('') + '\nAbstract: ' + df['abstract'].fillna('')
    
    # Define label mapping for binary classification
    label2id = {
        'Y': 1,  # BioNLP
        'N': 0   # Non-BioNLP
    }
    
    # Encode the labels
    df['label'] = df['isBionlp'].map(label2id)
    
    # Clean the data by selecting only the necessary columns
    df = df[['text', 'label']].reset_index(drop=True)
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return Dataset.from_pandas(df), label2id


def tokenize_dataset(dataset, tokenizer_name):
    """Tokenize the dataset"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Preprocess function
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return tokenized_dataset, tokenizer, data_collator

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average='macro')
    prec = precision.compute(predictions=predictions, references=labels, average='macro')
    rec = recall.compute(predictions=predictions, references=labels, average='macro')
    
    f1_classes = f1.compute(predictions=predictions, references=labels, average=None)
    f1_non_bionlp = f1_classes['f1'][0]
    f1_bionlp = f1_classes['f1'][1]
    
    return {
        'accuracy': acc['accuracy'],
        'f1': f1_score['f1'],
        'precision': prec['precision'],
        'recall': rec['recall'],
        'f1_non_bionlp': f1_non_bionlp,
        'f1_bionlp': f1_bionlp
    }

def main(model_name, train_file, eval_file, output_dir):
    # Load and tokenize training data
    train_dataset_raw, label2id = load_and_preprocess_data(train_file)
    train_dataset_tokenized, tokenizer, data_collator = tokenize_dataset(train_dataset_raw, model_name)
    train_dataset_tokenized = train_dataset_tokenized.remove_columns(['text'])
    
    # Optionally load and tokenize evaluation data
    eval_dataset_tokenized = None
    if eval_file:
        eval_dataset_raw, _ = load_and_preprocess_data(eval_file)
        eval_dataset_tokenized, _, _ = tokenize_dataset(eval_dataset_raw, model_name)
        eval_dataset_tokenized = eval_dataset_tokenized.remove_columns(['text'])
    
    # Mapping
    id2label = {v: k for k, v in label2id.items()}
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch" if eval_dataset_tokenized else "no",
        save_strategy="epoch" if eval_dataset_tokenized else "no",
        logging_steps=100,
        load_best_model_at_end=bool(eval_dataset_tokenized),
        metric_for_best_model="f1" if eval_dataset_tokenized else None,
        report_to=None
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_dataset_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset_tokenized else None
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary classification model for BioNLP.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="The name of the pre-trained model.")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to the training JSONL file.")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="(Optional) Path to the evaluation JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory to save model and tokenizer.")
    args = parser.parse_args()
    main(args.model_name, args.train_data, args.eval_data, args.output_dir)
