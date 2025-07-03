import json
import pandas as pd
from datasets import Dataset
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate
from tqdm import tqdm

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


def compute_metrics(predictions, labels):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average='macro')
    prec = precision.compute(predictions=predictions, references=labels, average='macro')
    rec = recall.compute(predictions=predictions, references=labels, average='macro')
    
    # Compute per-class metrics
    f1_classes = f1.compute(predictions=predictions, references=labels, average=None)
    f1_non_bionlp = f1_classes['f1'][0]  # index 0 corresponds to 'Non_BioNLP'
    f1_bionlp = f1_classes['f1'][1]      # index 1 corresponds to 'BioNLP'
    
    return {
        'accuracy': acc['accuracy'],
        'f1': f1_score['f1'],
        'precision': prec['precision'],
        'recall': rec['recall'],
        'f1_non_bionlp': f1_non_bionlp,
        'f1_bionlp': f1_bionlp
    }

def evaluate_model(model_path, eval_dataset, batch_size=16):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    def preprocess_function(examples, tokenizer):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(
        [col for col in tokenized_eval_dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(tokenized_eval_dataset, batch_size=batch_size, collate_fn=data_collator)

    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != 'label'})
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            labels.extend(batch['labels'].cpu().numpy())
            
    return compute_metrics(predictions, labels)


def main(model_path, jsonl_file):
    # Load and preprocess data
    eval_dataset, _ = load_and_preprocess_data(jsonl_file)
    
    # Evaluate the model
    metrics = evaluate_model(model_path, eval_dataset)
    
    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a BioNLP classification model.")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the trained model directory")
    parser.add_argument("--jsonl_file", type=str, required=True, 
                      help="The JSONL file containing the dataset")
    
    args = parser.parse_args()
    main(args.model_path, args.jsonl_file)
