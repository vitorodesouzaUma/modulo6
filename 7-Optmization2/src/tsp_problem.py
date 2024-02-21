from genetic.problems import Problem
from genetic.chromosome import Chromosome
from genetic.problems_types import ProblemType, ProblemDataType
import numpy as np


class TSPProblem(Problem):
    '''
    Class to implement TSP objective function.
    Inherits from Problem abstract class.
    '''

    problem_type: ProblemType = ProblemType.COMBINATORIAL
    problem_data_type: ProblemDataType = ProblemDataType.CATEGORICAL
    problem_description: str = "Minimize distance in Travelling salesman problem (TSP) problems" 

    def __init__(self, tsp_list: list):
        self.categories = tsp_list    
        super().__init__()
    
    def calculate_fitness(self, chromossome: Chromosome, *args, **kwargs):
        genes = chromossome.genes
        fitness = 0
        for i in range(len(genes)):
            if i < len(genes)-1:
                # Calculate distance to the next city
                fitness += self.calculate_distance(genes[i].value, genes[i+1].value)
            else:
                # Calculate distance from the last city to the first one
                fitness += self.calculate_distance(genes[i].value, genes[0].value)

        return 1/fitness # Maximize distance
    
    def calculate_distance(self, city1, city2):
        # Function to calculate distance between two cities
        distance = 0
        xDis = abs(city1[0] - city2[0])
        yDis = abs(city1[1] - city2[1])
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance