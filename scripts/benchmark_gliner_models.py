import argparse
import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm

def benchmark_models(input_file: str, output_file: str):
    """
    Évalue les modèles GLINER sur le dataset 'camembert/wikiner-fr'.

    Args:
        input_file (str): Chemin vers le fichier JSON contenant les IDs des modèles.
        output_file (str): Chemin vers le fichier de sortie JSON avec les résultats des benchmarks.
    """
    with open(input_file, 'r') as f:
        model_ids = json.load(f)

    # Charger le dataset
    dataset = load_dataset('camembert/wikiner-fr', split='test')  # Utilisez 'test' pour l'évaluation

    # Initialiser les métriques
    metrics = {
        'entity_f1': evaluate.load("seqeval"),
        'precision': evaluate.load("precision"),
        'recall': evaluate.load("recall")
    }

    results = []

    for model_id in tqdm(model_ids, desc="Benchmarking models"):
        try:
            # Charger le modèle et le tokenizer
            model = AutoModelForTokenClassification.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            predictions = []
            references = []

            for example in tqdm(dataset, desc=f"Evaluating {model_id}", leave=False):
                tokens = example['tokens']
                true_tags = example['ner_tags']

                inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=2).squeeze().tolist()

                # Convertir les IDs en labels
                predicted_labels = [model.config.id2label[pred_id] for pred_id in predicted_ids]
                true_labels = [model.config.id2label[tag_id] for tag_id in true_tags]

                predictions.append(predicted_labels)
                references.append(true_labels)

            # Calculer les métriques
            eval_results = {
                'model_id': model_id,
                'entity_f1': metrics['entity_f1'].compute(predictions=predictions, references=references)['overall_f1'],
                'precision': metrics['precision'].compute(predictions=predictions, references=references)['precision'],
                'recall': metrics['recall'].compute(predictions=predictions, references=references)['recall'],
                'timestamp': datetime.now().isoformat()
            }

            results.append(eval_results)
            print(f"Completed benchmarking for model: {model_id}")

        except Exception as e:
            print(f"Error benchmarking {model_id}: {e}")
            continue

    # Enregistrer les résultats
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Benchmarking completed. Results saved to {output_file}.")

if __name__ == '__main__':
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Benchmark GLINER models.")
    parser.add_argument('--input', type=str, required=True, help='Chemin vers le fichier JSON avec les IDs des modèles.')
    parser.add_argument('--output', type=str, required=True, help='Chemin vers le fichier de sortie JSON des résultats.')

    args = parser.parse_args()

    benchmark_models(args.input, args.output)
