import argparse
import json
from huggingface_hub import HfApi, ModelFilter

def discover_models(output_file: str, max_models: int = 50, force_update: bool = False):
    """
    Découvre les modèles GLINER sur Hugging Face Hub et les enregistre dans un fichier JSON.

    Args:
        output_file (str): Chemin vers le fichier de sortie JSON.
        max_models (int): Nombre maximum de modèles à récupérer.
        force_update (bool): Ignorer le cache et forcer la mise à jour.
    """
    api = HfApi()
    filter_kwargs = ModelFilter(
        model_name="gliner",
        task="token-classification",
        language="fr"
    )

    models = api.list_models(
        filter=filter_kwargs,
        sort="downloads",
        direction=-1,
        limit=max_models
    )

    model_ids = [model.modelId for model in models]

    with open(output_file, 'w') as f:
        json.dump(model_ids, f, indent=2)

    print(f"Discovered {len(model_ids)} GLINER models.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Discover GLINER models from Hugging Face Hub.")
    parser.add_argument('--output', type=str, required=True, help='Chemin vers le fichier de sortie JSON.')
    parser.add_argument('--max-models', type=int, default=50, help='Nombre maximum de modèles à récupérer.')
    parser.add_argument('--force-update', action='store_true', help='Forcer la mise à jour même si les données sont déjà à jour.')

    args = parser.parse_args()

    discover_models(args.output, args.max_models, args.force_update)
