import os
import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

import huggingface_hub
import torch
import datasets
import evaluate
from huggingface_hub import HfApi, ModelFilter
from tqdm import tqdm

class GLINERLeaderboard:
    def __init__(self, 
                 config_path: str = 'leaderboard_config.yaml',
                 output_dir: str = 'leaderboard_results'):
        """
        Initialize the GLINER Leaderboard manager
        
        Args:
            config_path (str): Path to configuration YAML
            output_dir (str): Directory to store leaderboard results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Hugging Face API setup
        self.hf_api = HfApi()
        
        # Evaluation metrics setup
        self.metrics = {
            'entity_f1': evaluate.load("seqeval"),
            'precision': evaluate.load("precision"),
            'recall': evaluate.load("recall")
        }
    
    def _filter_gliner_models(self) -> List[str]:
        """
        Retrieve GLINER models from Hugging Face Hub
        
        Returns:
            List of model repository names
        """
        filter_kwargs = {
            'filter_repos': ModelFilter(
                model_name='gliner',
                task='token-classification',
                language='fr'
            ),
            'sort': 'downloads',
            'direction': -1,
            'limit': self.config.get('max_models', 50)
        }
        
        models = self.hf_api.list_models(**filter_kwargs)
        return [model.modelId for model in models]
    
    def _evaluate_model(self, model_id: str) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            model_id (str): Hugging Face model repository ID
        
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load model and tokenizer
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            
            model = AutoModelForTokenClassification.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load French NER dataset
            fr_ner_dataset = datasets.load_dataset('camembert/wikiner-fr', split='train')
            
            # Evaluation logic
            predictions, references = [], []
            for example in tqdm(fr_ner_dataset):
                inputs = tokenizer(example['tokens'], is_split_into_words=True, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = model(**inputs).logits
                    predicted_label_ids = torch.argmax(outputs, dim=2).numpy()[0]
                
                predictions.append([model.config.id2label[pred] for pred in predicted_label_ids])
                references.append(example['ner_tags'])
            
            # Compute metrics
            results = {
                'entity_f1': self.metrics['entity_f1'].compute(predictions=predictions, references=references),
                'precision': self.metrics['precision'].compute(predictions=predictions, references=references),
                'recall': self.metrics['recall'].compute(predictions=predictions, references=references)
            }
            
            return {
                'model_id': model_id,
                **results,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"Error evaluating {model_id}: {e}")
            return None
    
    def update_leaderboard(self):
        """
        Main method to update GLINER model leaderboard
        """
        # Find GLINER models
        gliner_models = self._filter_gliner_models()
        
        # Evaluate models
        evaluation_results = []
        for model_id in gliner_models:
            result = self._evaluate_model(model_id)
            if result:
                evaluation_results.append(result)
        
        # Create DataFrame and sort
        leaderboard_df = pd.DataFrame(evaluation_results)
        leaderboard_df.sort_values(by='entity_f1', ascending=False, inplace=True)
        
        # Save to CSV
        leaderboard_path = os.path.join(
            self.output_dir, 
            f'gliner_leaderboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        leaderboard_df.to_csv(leaderboard_path, index=False)
        
        # Optional: Publish to Hugging Face Space
        if self.config.get('publish_to_space', False):
            self._publish_to_space(leaderboard_path)
    
    def _publish_to_space(self, leaderboard_path: str):
        """
        Publish leaderboard to Hugging Face Space
        
        Args:
            leaderboard_path (str): Path to leaderboard CSV
        """
        try:
            # Implementation for space publishing
            # Requires additional Hugging Face Space setup
            pass
        except Exception as e:
            print(f"Space publishing error: {e}")
    
    def continuous_update(self, interval_hours: int = 24):
        """
        Continuous leaderboard update loop
        
        Args:
            interval_hours (int): Hours between updates
        """
        while True:
            self.update_leaderboard()
            time.sleep(interval_hours * 3600)

# Configuration template
DEFAULT_CONFIG = {
    'max_models': 50,
    'publish_to_space': False,
    'metrics': ['entity_f1', 'precision', 'recall']
}

# Save default configuration if not exists
if not os.path.exists('leaderboard_config.yaml'):
    with open('leaderboard_config.yaml', 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f)

# Main execution
if __name__ == '__main__':
    leaderboard = GLINERLeaderboard()
    leaderboard.continuous_update()
