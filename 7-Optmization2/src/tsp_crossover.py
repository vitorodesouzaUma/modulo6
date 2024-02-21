import random

from genetic.crossovers import Crossover
from genetic.individual import Individual
from genetic.gene import Gene
from genetic.chromosome import Chromosome


class TSPCrossover(Crossover):

    def __init__(self):
        pass

    def crossover(
        self, parent1: Individual, parent2: Individual, *args, **kwargs
    ) -> Individual:
        
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
        #child_genes = [Gene(value=item) for item in child_list]
        child_chromosome = Chromosome(genes=child_list)

        return Individual(child_chromosome)