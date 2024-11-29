name: GLINER Model Leaderboard CICD

on:
  schedule:
    # Exécution quotidienne à minuit
    - cron: '0 0 * * *'
  workflow_dispatch:
    inputs:
      force_update:
        description: 'Force une mise à jour manuelle'
        required: false
        type: boolean
        default: false

env:
  PYTHONUNBUFFERED: 1
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
  SPACE_NAME: "gliner-leaderboard"

jobs:
  discover-and-benchmark:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Generate requirements.txt
      run: |
        pip freeze > requirements.txt
        cat requirements.txt

    - name: Discover New GLINER Models
      id: model-discovery
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        python scripts/discover_gliner_models.py \
          --output models_to_benchmark.json \
          --max-models 50 \
          --force-update ${{ github.event.inputs.force_update }}

    - name: Benchmark New Models
      id: benchmark
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        python scripts/benchmark_gliner_models.py \
          --input models_to_benchmark.json \
          --output benchmark_results.json

    - name: Update Leaderboard
      id: update-leaderboard
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        python scripts/update_leaderboard.py \
          --benchmark-results benchmark_results.json \
          --output-leaderboard leaderboard.csv

    - name: Commit and Push Changes
      run: |
        git config user.name 'GitHub Actions Bot'
        git config user.email '<>'
        git add leaderboard.csv requirements.txt
        git commit -m "Update GLINER Model Leaderboard" || exit 0
        git push

    - name: Update Hugging Face Space
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        huggingface-cli login --token $HF_TOKEN
        huggingface-cli repo create $SPACE_NAME --type space --sdk gradio
        cp leaderboard.csv space/$SPACE_NAME/
        cd space/$SPACE_NAME/
        git add leaderboard.csv
        git commit -m "Update Leaderboard Data"
        git push

# Scripts supplementaires à créer

# scripts/discover_gliner_models.py
```python
import argparse
import json
from huggingface_hub import HfApi, ModelFilter

def discover_models(max_models=50, force_update=False):
    api = HfApi()
    models = api.list_models(
        filter=ModelFilter(
            model_name='gliner', 
            task='token-classification', 
            language='fr'
        ),
        limit=max_models,
        sort='downloads'
    )
    
    model_ids = [model.modelId for model in models]
    
    with open('models_to_benchmark.json', 'w') as f:
        json.dump(model_ids, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-models', type=int, default=50)
    parser.add_argument('--force-update', type=bool, default=False)
    args = parser.parse_args()
    
    discover_models(args.max_models, args.force_update)
```

# scripts/benchmark_gliner_models.py
```python
import argparse
import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
import evaluate

def benchmark_models(models_file):
    with open(models_file, 'r') as f:
        model_ids = json.load(f)
    
    results = []
    metrics = {
        'f1': evaluate.load('seqeval'),
        'precision': evaluate.load('precision'),
        'recall': evaluate.load('recall')
    }
    
    dataset = load_dataset('camembert/wikiner-fr', split='train')
    
    for model_id in model_ids:
        try:
            model = AutoModelForTokenClassification.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Benchmark logic similar to previous implementation
            # Compute metrics and store results
            
            results.append({
                'model_id': model_id,
                'metrics': {}  # Populate with computed metrics
            })
        except Exception as e:
            print(f"Error benchmarking {model_id}: {e}")
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    benchmark_models(args.input)
```

# scripts/update_leaderboard.py
```python
import argparse
import json
import pandas as pd
from datetime import datetime

def update_leaderboard(benchmark_results):
    with open(benchmark_results, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    df.sort_values(by='f1_score', ascending=False, inplace=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['timestamp'] = timestamp
    
    df.to_csv('leaderboard.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-results', required=True)
    parser.add_argument('--output-leaderboard', required=True)
    args = parser.parse_args()
    
    update_leaderboard(args.benchmark_results)
```
