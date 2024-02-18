# -*- coding: utf-8 -*-
"""

    Module 6: Descriptive and Predictive Modeling
    Exercise 1: Optimization
    Submitted by: Jorge de la Torre Garcia (DNI), Lydia Phoebe Amanda Lilius (DNI), Miguel Galán Cisneros (DNI), Vitor Oliveira de Souza (Z0963220P)
    Date: 14/02/2024

"""

from itertools import product
import random
import math
from matplotlib import pyplot as plt
import numpy as np

# =============================================================================
# Objective function
# =============================================================================


## Calculates the fitness of an individual
def apply_function(individual):
    x = individual["x"]
    y = individual["y"]
    # Calculate sums of squares of x and y values.
    firstSum = x**2.0 + y**2.0
    # Calculate cosine of 2*pi*x and cosine of 2*pi*y.
    secondSum = math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)
    # n is the number of values (x and y = 2)
    n = 2
    # Returns a float value calculated by the Ackley function (objective function)
    return -(
        -20.0 * math.exp(-0.2 * math.sqrt(firstSum / n))
        - math.exp(secondSum / n)
        + 20
        + math.e
    )


# =============================================================================
# Generate first population
# =============================================================================


## Generates a population of individuals with random x and y values in the given boundaries
def generate_population(size, x_boundaries, y_boundaries):
    lower_x_boundary, upper_x_boundary = x_boundaries
    lower_y_boundary, upper_y_boundary = y_boundaries

    population = []
    for i in range(size):
        individual = {
            "x": random.uniform(lower_x_boundary, upper_x_boundary),
            "y": random.uniform(lower_y_boundary, upper_y_boundary),
        }
        population.append(individual)

    return population


# =============================================================================
# Selection Methods
# =============================================================================


## Select an individual from a popultion based on its fitness
def select_by_roulette(sorted_population, fitness_sum):
    # Initializes an offset to zero
    offset = 0
    # Initializes normalized fitness sum as fitness sum
    normalized_fitness_sum = fitness_sum

    # If the lowest fitness is negative, we add the absolute value of the
    # lowest fitness multiplied by the number of individuals in the population
    # to the normalized fitness sum
    lowest_fitness = apply_function(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)

    # Gets an random number between 0 and 1
    draw = random.uniform(0, 1)

    # Initializes the probability threashold to zero
    accumulated = 0
    # Test each individual in the sorted polulation (from the lowest to the highest fitness)
    # and then calculate the accumulated probability until it is bigger than the
    # draw value randomly set in the previous step
    for individual in sorted_population:
        fitness = apply_function(individual) + offset
        # Increasing probability
        # It never takes the worst individual since its probability is always zero
        probability = fitness / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            return individual


## Select randomly
def select_random(population):
    return random.choice(population)


# =============================================================================
# Sort Population method
# =============================================================================


## Sorts the population by fitness in ascending order
def sort_population_by_fitness(population):
    return sorted(population, key=apply_function)


# =============================================================================
# Crossover Method
# =============================================================================


## Create a new individual based on the mean of x and y values of its parents
def crossover(individual_a, individual_b):
    xa = individual_a["x"]
    ya = individual_a["y"]

    xb = individual_b["x"]
    yb = individual_b["y"]

    # Return one dictionary with x and y as the mean of the two individuals sent as parameters
    return {"x": (xa + xb) / 2, "y": (ya + yb) / 2}


# =============================================================================
# Mutation Method
# =============================================================================


def random_mutation(lower_boundry, upper_boundry, value, mutation_value):

    boundries_diff = upper_boundry - lower_boundry
    # mutation is done by adding a normally distributed random number to the gene's value
    # Keep values in boundry limits
    new_value = min(
        max(
            value
            + random.uniform(
                -boundries_diff * mutation_value, boundries_diff * mutation_value
            ),
            lower_boundry,
        ),
        upper_boundry,
    )

    return new_value


## Mutate the individual
def mutate(individual, mutation_value, evolution_percantage):
    # Sets the boundry values for x and y
    lower_boundary, upper_boundary = (-4, 4)

    if mutation_value == 'auto':
        if evolution_percantage < 0.3:
            mutation_value = 0.2
        elif evolution_percantage < 0.7:
            mutation_value = 0.1
        else:
            mutation_value = 0.05

    next_x = random_mutation(
        lower_boundary, upper_boundary, individual["x"], mutation_value=mutation_value
    )
    next_y = random_mutation(
        lower_boundary, upper_boundary, individual["y"], mutation_value=mutation_value
    )

    # Return one dictionary with x and y as the new mutated values (not exceeding the boundries specified)
    return {"x": next_x, "y": next_y}


# =============================================================================
# Make Next Generation
# =============================================================================


## Improved version of the function to create th next generation
## In this version we only append the new individuals that have a
## better fitness than the worst individual in the previous generation
def make_next_generation(previous_population, selection_method, mutation_value, evolution_percantage):
    # Sorts population by fitness in ascending order
    # (the higher the fitness, the better the individual)
    sorted_by_fitness_population = sort_population_by_fitness(previous_population)
    # Gets population size
    population_size = len(previous_population)

    # Creates a new individual for each individual in the previous population
    # and replace one individual with the best individual created
    for i in range(population_size):
        # Gets the sum of all fitnesses
        fitness_sum = sum(
            apply_function(individual) for individual in sorted_by_fitness_population
        )
        # Randomly select two individuals from the sorted population
        if selection_method == "roulette":
            father = select_by_roulette(sorted_by_fitness_population, fitness_sum)
            mother = select_by_roulette(sorted_by_fitness_population, fitness_sum)
        elif selection_method == "random":
            father = select_random(sorted_by_fitness_population)
            mother = select_random(sorted_by_fitness_population)

        # Creates a new individual by crossing the two selected individuals
        individual = crossover(father, mother)

        # Mutates the new individual
        individual = mutate(individual, mutation_value, evolution_percantage)

        if apply_function(individual) > apply_function(sorted_by_fitness_population[0]):
            # Replace the worst individual in the previous generation with the new individual
            sorted_by_fitness_population[0] = individual

    # Return a list with new individuals (next generation)
    return sorted_by_fitness_population


# =============================================================================
# Genetic Algorithm
# =============================================================================


def genetic_algorithm(
    selection_method="random",
    graph=False,
    mutation_value=0.1,
    verbose=0,
):
    generations = 100
    population = generate_population(
        size=10, x_boundaries=(-5, 5), y_boundaries=(-5, 5)
    )

    i = 1
    bestFitness = []
    while True:

        if verbose > 0:
            print(str(i))

        for individual in population:
            if verbose > 0:
                print(individual, apply_function(individual))

        if i == generations:
            break

        i += 1

        population = make_next_generation(
            previous_population=population,
            selection_method=selection_method,
            mutation_value=mutation_value,
            evolution_percantage= (i / generations)
        )
        best_individual = sort_population_by_fitness(population)[-1]
        bestFitness.append(apply_function(best_individual))

    best_individual = sort_population_by_fitness(population)[-1]

    if graph:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(bestFitness)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title("Genetic Algorithm Performance History")
        plt.show()

    if verbose > 0:
        print("\nFINAL RESULT")
    if verbose > 0:
        print(best_individual, apply_function(best_individual))

    return best_individual, apply_function(best_individual)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    cross_validation = False

    if not cross_validation:
        genetic_algorithm(verbose=1, graph=True, mutation_value='auto')
    else:

        # Define parameter grid
        param_grid = {
            "selection_method": ["roulette", "random"],
            "mutation_value": [0.025, 0.05, 0.1, 0.2, 'auto'],
        }

        # Number of "cross validations" runs each combination of parameters
        n_cv = 200

        # Variable to store the cross validation results
        results = {}

        # Grid search loop
        for params in product(
            param_grid["selection_method"], param_grid["mutation_value"]
        ):
            selection_method, mutation_value = params
            best_individuals = []
            best_fitnesses = []

            # Cross validation loop
            for i in range(n_cv):
                best_individual, fitness = genetic_algorithm(
                    selection_method=selection_method,
                    mutation_value=mutation_value,
                )
                best_individuals.append(best_individual)
                best_fitnesses.append(fitness)

            # Store results for each parameters combination
            results[params] = {
                "best_individuals": best_individuals,
                "best_fitnesses": best_fitnesses,
            }

        # Plotting and printing results
        average_fitnesses = []
        std_devs = []
        n_local_minimas = []
        labels = []

        # Extract data for plotting
        for param, data in results.items():
            average_fitness = np.mean(data["best_fitnesses"])
            std_dev = np.std(data["best_fitnesses"])
            n_local_minima = sum(np.diff(data["best_fitnesses"]) < -1)
            print(
                f"Average solution for {param}: {average_fitness} with std: {std_dev} - Total local minima: {n_local_minima}"
            )

            average_fitnesses.append(average_fitness)
            std_devs.append(std_dev)
            n_local_minimas.append(n_local_minima)
            labels.append(f"{param[0]}-{param[1]}")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bar plot
        x_pos = np.arange(len(labels))
        bars = ax.bar(
            x_pos,
            average_fitnesses,
            yerr=std_devs,
            capsize=5,
            alpha=0.75,
            color="skyblue",
        )

        # Annotate bars with the count of local minima
        for bar, n_minima in zip(bars, n_local_minimas):
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                f"LM: {n_minima}",
                va="bottom",
            )  # LM: Local Minima

        ax.set_xlabel("Parameter Combination")
        ax.set_ylabel("Average Fitness")
        ax.set_title("Genetic Algorithm Performance")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")

        plt.tight_layout()
        plt.show()
