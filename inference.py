import json
import pandas as pd
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def load_data(jsonl_file):
    # Load the dataset from JSONL file
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
    
    # Filter for required fields
    filtered_data = []
    for item in data:
        if all(key in item for key in ['title', 'abstract']):
            filtered_data.append(item)
    
    return filtered_data

def predict(model_path, data):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Get id2label mapping from the model config
    id2label = model.config.id2label
    
    results = []
    
    for item in tqdm(data, desc="Processing"):
        # Prepare input text
        text = f"Title: {item.get('title', '')}\nAbstract: {item.get('abstract', '')}"
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        
        # Create result object
        result = {
            'paperId': item.get('paperId', ''),
            'title': item.get('title', ''),
            "abstract": item.get('abstract', ''),
            'predicted_label': id2label[prediction],
            'confidence': probabilities[0][prediction].item()
        }
        
        results.append(result)
    
    return results

def main(model_path, input_jsonl, output_jsonl):
    # Load data
    data = load_data(input_jsonl)
    print(f"Loaded {len(data)} papers for inference")
    
    # Make predictions
    results = predict(model_path, data)
    
    # Save results
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Predictions saved to {output_jsonl}")
    
    # Print summary
    predictions = [r['predicted_label'] for r in results]
    bionlp_count = predictions.count('Y')
    non_bionlp_count = predictions.count('N')
    
    print(f"Summary:")
    print(f"  BioNLP papers: {bionlp_count} ({bionlp_count/len(predictions)*100:.1f}%)")
    print(f"  Non-BioNLP papers: {non_bionlp_count} ({non_bionlp_count/len(predictions)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained BioNLP classification model.")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the trained model directory")
    parser.add_argument("--input_jsonl", type=str, required=True, 
                      help="The JSONL file containing papers to classify")
    parser.add_argument("--output_jsonl", type=str, required=True, 
                      help="The output JSONL file where predictions will be written")
    
    args = parser.parse_args()
    main(args.model_path, args.input_jsonl, args.output_jsonl)

  