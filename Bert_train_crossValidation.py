import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
import argparse
from sklearn.model_selection import KFold
import os
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

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
    
    return df, label2id

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
    """Compute evaluation metrics"""
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
    
    # Compute per-class metrics
    f1_classes = f1.compute(predictions=predictions, references=labels, average=None)
    f1_non_bionlp = f1_classes['f1'][0]  # index 0 corresponds to 'Non-BioNLP' (N)
    f1_bionlp = f1_classes['f1'][1]      # index 1 corresponds to 'BioNLP' (Y)
    
    return {
        'accuracy': acc['accuracy'],
        'f1': f1_score['f1'],
        'precision': prec['precision'],
        'recall': rec['recall'],
        'f1_non_bionlp': f1_non_bionlp,
        'f1_bionlp': f1_bionlp
    }

def train_single_fold(train_dataset, eval_dataset, model_name, label2id, output_dir, fold_num):
    """Train a single fold"""
    print(f"\n=== Training Fold {fold_num + 1} ===")
    
    # Id2label for binary classification
    id2label = {v: k for k, v in label2id.items()}
    
    # Load model for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        id2label=id2label, 
        label2id=label2id
    )
    
    # Check if CUDA is available and set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create fold-specific output directory
    fold_output_dir = os.path.join(output_dir, f"fold_{fold_num + 1}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{fold_output_dir}/logs",
        logging_strategy="epoch",
        save_total_limit=1,
        report_to=None  # Disable wandb/tensorboard logging
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    
    print(f"Fold {fold_num + 1} Results:")
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            print(f"  {key}: {value:.4f}")
    
    # Save the model
    trainer.save_model(fold_output_dir)
    tokenizer.save_pretrained(fold_output_dir)
    
    return eval_results

def main(model_name, jsonl_file, output_dir, n_splits=5):
    """Main function to run 5-fold cross-validation"""
    # Load and preprocess data
    df, label2id = load_and_preprocess_data(jsonl_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results from all folds
    all_results = []
    
    # Perform k-fold cross-validation
    for fold_num, (train_idx, eval_idx) in enumerate(kfold.split(df)):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold_num + 1}/{n_splits}")
        print(f"{'='*50}")
        
        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        eval_df = df.iloc[eval_idx].reset_index(drop=True)
        
        print(f"Train samples: {len(train_df)}")
        print(f"Eval samples: {len(eval_df)}")
        print(f"Train label distribution: {train_df['label'].value_counts().to_dict()}")
        print(f"Eval label distribution: {eval_df['label'].value_counts().to_dict()}")
        
        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        # Tokenize datasets
        train_tokenized, tokenizer, data_collator = tokenize_dataset(train_dataset, model_name)
        eval_tokenized, _, _ = tokenize_dataset(eval_dataset, model_name)
        
        # Remove text column (keep only input_ids, attention_mask, label)
        train_tokenized = train_tokenized.remove_columns(['text'])
        eval_tokenized = eval_tokenized.remove_columns(['text'])
        
        # Train the fold
        fold_results = train_single_fold(
            train_tokenized, eval_tokenized, model_name, 
            label2id, output_dir, fold_num
        )
        
        # Store results
        fold_results['fold'] = fold_num + 1
        all_results.append(fold_results)
    
    # Calculate and display overall statistics
    print(f"\n{'='*50}")
    print("OVERALL CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    
    # Aggregate results
    metrics = ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall', 
               'eval_f1_non_bionlp', 'eval_f1_bionlp']
    
    results_summary = {}
    for metric in metrics:
        values = [result[metric] for result in all_results]
        results_summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    # Print summary
    print("\nMetric Summary (Mean ± Std):")
    print("-" * 40)
    for metric, stats in results_summary.items():
        metric_name = metric.replace('eval_', '').replace('_', ' ').title()
        print(f"{metric_name:20}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(Range: {stats['min']:.4f} - {stats['max']:.4f})")
    
    # Print individual fold results
    print(f"\nIndividual Fold Results:")
    print("-" * 80)
    print(f"{'Fold':>4} {'Accuracy':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'F1-NonBio':>10} {'F1-Bio':>8}")
    print("-" * 80)
    
    for i, result in enumerate(all_results):
        print(f"{i+1:>4} {result['eval_accuracy']:>8.4f} {result['eval_f1']:>8.4f} "
              f"{result['eval_precision']:>10.4f} {result['eval_recall']:>8.4f} "
              f"{result['eval_f1_non_bionlp']:>10.4f} {result['eval_f1_bionlp']:>8.4f}")
    
    # Save results to file
    results_file = os.path.join(output_dir, "cross_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'individual_results': all_results,
            'summary': results_summary,
            'model_name': model_name,
            'n_splits': n_splits
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Model checkpoints saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary classification model for BioNLP using 5-fold cross-validation.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                      help="The name of the pre-trained model.")
    parser.add_argument("--jsonl_file", type=str, required=True,
                      help="The JSONL file containing the dataset.")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="The output directory where the model and checkpoints will be written.")
    parser.add_argument("--n_splits", type=int, default=5,
                      help="Number of folds for cross-validation (default: 5).")
    
    args = parser.parse_args()
    main(args.model_name, args.jsonl_file, args.output_dir, args.n_splits)