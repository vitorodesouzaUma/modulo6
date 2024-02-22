import math
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
        

        return Chromosome(genes=genes)
    

class DynamicSwapMutation(Mutation):

    def __init__(self, 
                 total_gen, 
                 pop_size, 
                 max_mutation_rate = 0.2, 
                 min_mutation_rate = 0.025,
                 how = 'linear',
                 k = 10^-3):
        
        super().__init__(max_mutation_rate)
        self.max_mutation_rate = max_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.total_gen = total_gen
        self.pop_size = pop_size
        self.step = (max_mutation_rate - min_mutation_rate) / (total_gen * pop_size)
        self.current_step = 0
        self.how = how
        self.k = k

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

        if self.how == 'linear':
            # Dynamically adjust mutation rate
            self.mutation_rate = self.mutation_rate - self.step
        elif self.how == 'exponential':
            # Dynamically adjust mutation rate using exponential decay
            self.current_step += 1  # Increment current step
            self.mutation_rate = self.min_mutation_rate + (self.max_mutation_rate - self.min_mutation_rate) * math.exp(self.k * self.current_step)
            # Ensure mutation rate does not fall below the minimum
            self.mutation_rate = max(self.mutation_rate, self.min_mutation_rate)

        return Chromosome(genes=genes)