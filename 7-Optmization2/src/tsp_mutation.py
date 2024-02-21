from genetic.mutations import Mutation
from genetic.individual import Individual
from genetic.chromosome import Chromosome

import random

class SwapMutation(Mutation):
    
    def __init__(self, mutation_rate):
        super().__init__(mutation_rate)

    def mutate(
        self,
        individual: Individual,
        *args,
        **kwargs
    ):
        
        genes = individual.chromosome.genes

        for swapped in range(len(genes)):
            if(random.random() < self.mutation_rate):
                swapWith = int(random.random() * len(genes))
                
                gene1 = genes[swapped]
                gene2 = genes[swapWith]
                
                genes[swapped] = gene2
                genes[swapWith] = gene1
        
        #individual.chromosome.genes = genes

        return Chromosome(genes=genes)