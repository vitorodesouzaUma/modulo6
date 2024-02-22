import math
import random

from genetic.crossovers import Crossover
from genetic.individual import Individual
from genetic.gene import Gene
from genetic.chromosome import Chromosome


class PartialMappedCrossover(Crossover):

    def __init__(self,crossover_rate):
        super().__init__(crossover_rate)

    def crossover(
        self, parent1: Individual, parent2: Individual, *args, **kwargs
    ) -> Individual:
            
        if (random.random() < self.crossover_rate): 
            # Empty lists for merging two chromossomes
            child_list = []
            childP1 = []
            childP2 = []
            
            # Get parents genes lists
            parent1_genes = parent1.chromosome.genes
            parent2_genes = parent2.chromosome.genes

            # Sets start and eng genes for parent 1
            geneA = int(random.random() * len(parent1_genes))
            geneB = int(random.random() * len(parent2_genes))
            startGene = min(geneA, geneB)
            endGene = max(geneA, geneB)

            # Fill list with parent1 genes
            for i in range(startGene, endGene):
                childP1.append(parent1_genes[i])
            
            # Fill list with parent2 genes
            childP2 = [item for item in parent2_genes if item not in childP1]

            # Merging two parents genes and creating a new chromossome
            child_list = childP1 + childP2
            child_chromosome = Chromosome(genes=child_list)

            return Individual(child_chromosome)
            
        else:
            return parent1
    

class DynamicPartialMappedCrossover(PartialMappedCrossover):

    def __init__(self, 
                 min_crossover_rate,
                 max_crossover_rate, 
                 total_gen, 
                 pop_size,
                 how='linear',
                 k = 10^-3):

        super().__init__(max_crossover_rate)
        self.min_crossover_rate = min_crossover_rate
        self.max_crossover_rate = max_crossover_rate
        self.total_gen = total_gen
        self.pop_size = pop_size
        self.step = (max_crossover_rate - min_crossover_rate) / (total_gen * pop_size)
        self.how = how
        self.k = k
        self.current_step = 0

    def crossover(
        self, parent1: Individual, parent2: Individual, *args, **kwargs
    ) -> Individual:

        if (random.random() < self.crossover_rate): 
            # Empty lists for merging two chromossomes
            child_list = []
            childP1 = []
            childP2 = []
            
            # Get parents genes lists
            parent1_genes = parent1.chromosome.genes
            parent2_genes = parent2.chromosome.genes

            # Sets start and eng genes for parent 1
            geneA = int(random.random() * len(parent1_genes))
            geneB = int(random.random() * len(parent2_genes))
            startGene = min(geneA, geneB)
            endGene = max(geneA, geneB)

            # Fill list with parent1 genes
            for i in range(startGene, endGene):
                childP1.append(parent1_genes[i])
            
            # Fill list with parent2 genes
            childP2 = [item for item in parent2_genes if item not in childP1]

            # Merging two parents genes and creating a new chromossome
            child_list = childP1 + childP2
            child_chromosome = Chromosome(genes=child_list)


            if self.how == 'linear':
                # Dynamically adjust mutation rate
                self.crossover_rate = self.crossover_rate - self.step
            elif self.how == 'exponential':
                # Dynamically adjust mutation rate using exponential decay
                self.current_step += 1  # Increment current step
                self.crossover_rate = self.min_crossover_rate + (self.max_crossover_rate - self.min_crossover_rate) * math.exp(self.k * self.current_step)
                # Ensure mutation rate does not fall below the minimum
                self.crossover_rate = max(self.crossover_rate, self.min_crossover_rate)

            return Individual(child_chromosome)
            
        else:
            return parent1
        
