#import genetic # General library created to implement Genetic Algorithms
from matplotlib import pyplot as plt
import numpy as np
from tsp_mutation import SwapMutation
from tsp_problem import TSPProblem
from tsp_crossover import TSPCrossover
from tsp_parser import parse_tsp
from datetime import datetime
import json
import os
import pandas as pd

from genetic.genetic import Genetic
from genetic.selections import RouletteSelection
from genetic.utils.utils import plot_history

EXPERIMENT_FOLDER = "C:\\Users\\vos_v\\OneDrive\\Documentos\\00 - Europa\\00 - Master\\Módulo 4\\00-Jupyter\\Modulo6\\7-Optmization2\\experiments\\"

def sanitize_filename(filename):
    """Replaces invalid file name characters in a string with an underscore."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def save_ga_parameters(ga_parameters, folder, experiment_name):
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{experiment_name}_param.json")
    with open(filename, 'w') as f:
        json.dump(ga_parameters, f, indent=4)


def save_history(distance_history, ga_parameters, folder, experiment_name):
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{experiment_name}_hist.png")
    
    plt.figure(figsize=(15, 8)) 
    
    # Plot the distance history
    plt.plot(distance_history)
    plt.title('TSP Solution Distance over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.grid(True)
    
    # Convert GA parameters to a formatted string and add to the plot
    params_text = "\n".join(f"{key}: {value}" for key, value in ga_parameters.items())
    plt.gcf().text(0.7, 0.82, params_text, fontsize=9, verticalalignment='top',horizontalalignment='left',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.savefig(filename) 
    plt.close()


if __name__ == "__main__":    

    # Parse .tsp file
    # The function parse_tsp was developed to return a pandas DataFrame (it seems more genetical)
    data = parse_tsp(r"Modulo6\7-Optmization2\src\data\berlin52.tsp")
    # Create DataFrame
    df = pd.DataFrame(data['DATA']).set_index(0)
    # Set column's names
    df.columns = ["x", "y"]
    # Convert dtypes to float (it came as text from the file)
    df = df.astype(float)
    # Convert to list of tupples to use in genetic algorithm
    tsp_list = list(df.itertuples(index=False, name=None))

    

    print('\n******************** TSP solver ********************')
    print('\nDATASET INFO: \n')
    print('\t',data['NAME'])
    print('\t',data['DIMENSION'])
    print('\t',data['EDGE_WEIGHT_TYPE'])
    print('\n****************************************************')

    
    ga = Genetic(
        population_size=50,
        generations=10000,
        crossover_rate=0.5,
        problem=TSPProblem(tsp_list=tsp_list),
        crossover=TSPCrossover(),
        selection=RouletteSelection(),
        mutation=SwapMutation(mutation_rate=0.1),
        verbose=1,
    )

    start = datetime.now()
    history, best_individual = ga.run()
    end = datetime.now() - start
    print(f"\nTime: {end}\n")

    distance_history = [1/individual.fitness for individual in history]
    best_distance = 1/best_individual.get_fitness()

    print("Best distance: ", best_distance)

    dataset_name = data['NAME'].replace('NAME: ', '')
    # Sanitize and shorten the experiment name if necessary
    experiment_name = sanitize_filename(f"{dataset_name}_{round(best_distance,4)}_{datetime.now().strftime('%H-%M-%S')}")

    # Define the folder path
    experiment_folder = os.path.join(EXPERIMENT_FOLDER, experiment_name)

    ga_parameters = {
        "population_size": ga.population_size,
        "generations": ga.generations,
        "crossover_rate": ga.crossover_rate,
        "problem_class": ga.problem.__class__.__name__,
        "crossover_class": ga.crossover.__class__.__name__,
        "selection_class": ga.selection.__class__.__name__,
        "mutation_class": ga.mutation.__class__.__name__,
        "mutation_rate": ga.mutation.mutation_rate if hasattr(ga.mutation, 'mutation_rate') else None,
    }
    
    # Save the plot and parameters
    save_history(distance_history, ga_parameters, experiment_folder, experiment_name)
    save_ga_parameters(ga_parameters, experiment_folder, experiment_name)
    