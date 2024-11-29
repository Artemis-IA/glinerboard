import argparse
import json
import pandas as pd
from datetime import datetime

def update_leaderboard(benchmark_results_file: str, output_leaderboard: str):
    """
    Met à jour le fichier leaderboard CSV avec les résultats des benchmarks.

    Args:
        benchmark_results_file (str): Chemin vers le fichier JSON des résultats des benchmarks.
        output_leaderboard (str): Chemin vers le fichier de sortie CSV du leaderboard.
    """
    with open(benchmark_results_file, 'r') as f:
        results = json.load(f)

    # Convertir en DataFrame
    df = pd.DataFrame(results)

    # Trier par 'entity_f1' décroissant
    df.sort_values(by='entity_f1', ascending=False, inplace=True)

    # Ajouter une colonne de timestamp si nécessaire
    df['timestamp'] = datetime.now().isoformat()

    # Enregistrer en CSV
    df.to_csv(output_leaderboard, index=False)

    print(f"Leaderboard updated and saved to {output_leaderboard}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Update GLINER Model Leaderboard.")
    parser.add_argument('--benchmark-results', type=str, required=True, help='Chemin vers le fichier JSON des résultats des benchmarks.')
    parser.add_argument('--output-leaderboard', type=str, required=True, help='Chemin vers le fichier de sortie CSV du leaderboard.')

    args = parser.parse_args()

    update_leaderboard(args.benchmark_results, args.output_leaderboard)
