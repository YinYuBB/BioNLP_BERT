import json
import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm
import evaluate

def load_and_preprocess_data(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print("Warning: Skipping invalid JSON line")

    filtered = []
    for item in data:
        if all(k in item for k in ['title', 'abstract', 'isBionlp']) and item['isBionlp'] in ['Y', 'N']:
            filtered.append(item)

    df = pd.DataFrame(filtered)
    df['text'] = 'Title: ' + df['title'].fillna('') + '\nAbstract: ' + df['abstract'].fillna('')
    label2id = {'Y': 1, 'N': 0}
    df['label'] = df['isBionlp'].map(label2id)
    df = df[['text', 'label']].reset_index(drop=True)
    return df

def compute_metrics(preds, labels):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    acc = accuracy.compute(predictions=preds, references=labels)
    f1_macro = f1.compute(predictions=preds, references=labels, average='macro')
    prec = precision.compute(predictions=preds, references=labels, average='macro')
    rec = recall.compute(predictions=preds, references=labels, average='macro')
    f1_per_class = f1.compute(predictions=preds, references=labels, average=None)

    return {
        "accuracy": acc['accuracy'],
        "f1": f1_macro['f1'],
        "precision": prec['precision'],
        "recall": rec['recall'],
        "f1_non_bionlp": f1_per_class['f1'][0],
        "f1_bionlp": f1_per_class['f1'][1]
    }

def evaluate_on_fold(model_path, eval_df, batch_size=16):
    print(f"Eval label distribution: {eval_df['label'].value_counts().to_dict()}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataset = Dataset.from_pandas(eval_df)
    dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding="max_length", max_length=512), batched=True)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    preds, labels = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = batch['labels'].cpu().numpy()

            preds.extend(batch_preds)
            labels.extend(batch_labels)

    return compute_metrics(preds, labels)

def main(model_path, jsonl_file, output_dir, n_splits):
    df = load_and_preprocess_data(jsonl_file)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (_, test_idx) in enumerate(kf.split(df)):
        print(f"\n===== Evaluating Fold {fold+1}/{n_splits} =====")
        test_df = df.iloc[test_idx].reset_index(drop=True)
        fold_metrics = evaluate_on_fold(model_path, test_df)
        fold_metrics['fold'] = fold + 1
        results.append(fold_metrics)

        for k, v in fold_metrics.items():
            if k != 'fold':
                print(f"{k}: {v:.4f}")

    # Summarize results
    print("\n===== Cross-Validation Summary =====")
    if len(results) == 0:
        print("No evaluation results to summarize. Check if model path is correct.")
        return

    summary = {}
    for key in results[0].keys():
        if key == 'fold': continue
        values = [r[key] for r in results]
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
        print(f"{key:20}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f} (min: {summary[key]['min']:.4f}, max: {summary[key]['max']:.4f})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cross_validation_eval_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "fold_results": results,
            "summary": summary,
            "model_path": model_path,
            "n_splits": n_splits
        }, f, indent=2)

    print(f"\nSaved cross-validation results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validation evaluation for a single BioNLP model on each fold's test split.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to a trained model to evaluate on all folds' test sets")
    parser.add_argument("--jsonl_file", type=str, required=True, help="Evaluation data in JSONL format")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store results")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for K-Fold CV")
    args = parser.parse_args()
    main(args.model_path, args.jsonl_file, args.output_dir, args.n_splits)
