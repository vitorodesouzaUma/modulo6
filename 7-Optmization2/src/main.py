from itertools import product
from datetime import datetime
import os
import argparse

import pandas as pd

from tsp_mutation import DynamicSwapMutation, SwapMutation
from tsp_problem import TSPProblem
from tsp_crossover import PartialMappedCrossover
from tsp_utils import (
    parse_tsp,
    sanitize_filename,
    save_ga_parameters,
    save_gscv_hist,
    save_history,
    plot_path,
    save_solution,
)

from genetic.genetic import Genetic
from genetic.selections import RouletteSelection

EXPERIMENT_FOLDER = "Modulo6\\7-Optmization2\\experiments\\"


def solve(
    data,
    population_size,
    generations,
    selection,
    crossover,
    mutation,
    cross_validation_folder="",
    verbose=1,
):
    print("\n******************** TSP solver ********************")
    print("\nDATASET INFO: \n")
    print("\t", data["NAME"])
    print("\t", data["DIMENSION"])
    print("\t", data["EDGE_WEIGHT_TYPE"])
    print("\n****************************************************\n")

    ga = Genetic(
        population_size=population_size,
        generations=generations,
        problem=TSPProblem(tsp_list=tsp_list),
        crossover=crossover,
        selection=selection,
        mutation=mutation,
        verbose=verbose,
    )

    start = datetime.now()
    history, best_individual = ga.run()
    elapsed_time = datetime.now() - start
    print(f"\nTotal elapsed ime: {elapsed_time}\n")

    distance_history = [1 / individual.fitness for individual in history]
    best_distance = 1 / best_individual.get_fitness()

    print("Best distance: ", best_distance)

    dataset_name = data["NAME"].replace("NAME", "")
    dataset_name = dataset_name.replace(":", "").strip()
    # Sanitize and shorten the experiment name if necessary
    experiment_name = sanitize_filename(
        f"{dataset_name}_{round(best_distance,4)}_{datetime.now().strftime('%H-%M-%S')}"
    )

    # Define the folder path
    experiment_folder = os.path.join(
        EXPERIMENT_FOLDER, cross_validation_folder, experiment_name
    )

    ga_parameters = {
        "population_size": ga.population_size,
        "generations": ga.generations,
        "crossover_rate": ga.crossover.crossover_rate,
        "problem_class": ga.problem.__class__.__name__,
        "crossover_class": ga.crossover.__class__.__name__,
        "selection_class": ga.selection.__class__.__name__,
        "mutation_class": ga.mutation.__class__.__name__,
        "mutation_rate": ga.mutation.mutation_rate,
        "Total computational time": str(elapsed_time),
    }

    # Save the plot and parameters
    save_history(distance_history, ga_parameters, experiment_folder, experiment_name)
    save_ga_parameters(ga_parameters, experiment_folder, experiment_name)
    save_solution(best_individual.chromosome.genes, experiment_folder, experiment_name)

    solution_path = [[gene.value[0] for gene in best_individual.chromosome.genes]]
    solution_path.append([gene.value[1] for gene in best_individual.chromosome.genes])
    plot_path(solution_path, experiment_folder, experiment_name)

    return best_distance


def arg_parser():
    parser = argparse.ArgumentParser(description="TSP Genetic Algorithm")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the .tsp file",
        # default="Modulo6\\7-Optmization2\\src\\data\\bier127.tsp",
        default="Modulo6\\7-Optmization2\\src\\data\\a280.tsp",
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        help="Number of generations",
        default=100000,
    )
    args = parser.parse_args()
    return args.file, args.generations


# ===========================================================================================
# MAIN
# ===========================================================================================


if __name__ == "__main__":
    # add argparser to get the file path
    FILE_PATH, GENERATIONS = arg_parser()

    # Parse .tsp file
    # The function parse_tsp was developed to return a pandas DataFrame (it seems more genetical)
    data = parse_tsp(FILE_PATH)
    # Create DataFrame
    df = pd.DataFrame(data["DATA"]).set_index(0)
    # Set column's names
    df.columns = ["x", "y"]
    # Convert dtypes to float (it came as text from the file)
    df = df.astype(float)
    # Convert to list of tupples to use in genetic algorithm
    tsp_list = list(df.itertuples(index=False, name=None))

    cross_validation = False

    if not cross_validation:
        population_size = 80
        generations = GENERATIONS
        selection = RouletteSelection()
        crossover = PartialMappedCrossover(crossover_rate=1)
        mutation = SwapMutation(mutation_rate=0.05)

        best = solve(
            data,
            population_size,
            generations,
            selection,
            crossover,
            mutation,
            verbose=0,
        )

    else:
        cross_validation_folder = f"cv_{str(datetime.now().strftime('%H-%M-%S'))}"

        # Define parameter grid
        param_grid = {
            "mutation_rate": [0.1, 0.05],
            "population_size": [50, 80],
            "generations": [15000],
        }

        # Number of "cross validations" runs each combination of parameters
        n_cv = 5

        # Variable to store the cross validation results
        results = {}

        # Grid search loop
        for params in product(
            param_grid["mutation_rate"],
            param_grid["population_size"],
            param_grid["generations"],
        ):
            mutation_rate, population_size, generations = params
            best_individuals = []
            best_fitnesses = []

            selection = RouletteSelection()
            crossover = PartialMappedCrossover(crossover_rate=1)
            mutation = SwapMutation(mutation_rate=mutation_rate)

            # Cross validation loop
            for i in range(n_cv):
                fitness = solve(
                    data,
                    population_size,
                    generations,
                    selection,
                    crossover,
                    mutation,
                    cross_validation_folder,
                    verbose=0,
                )
                best_fitnesses.append(fitness)

            # Store results for each parameters combination
            results[params] = {
                "best_fitnesses": best_fitnesses,
            }

        save_gscv_hist(results, cross_validation_folder)
