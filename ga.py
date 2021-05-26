import sys
import os
import pygad
import tensorflow as tf
import numpy as np

no_of_generations = 15
no_of_individuals = 10
mutate_factor = 0.1
individuals = []

def mutate(new_individual):
  genes = []
  for gene in new_individual:
    n = random.random()
    if(n < mutate_factor):
      #Assign random values to certain genes within the maximum acceptable bounds
      genes.append(random.random())
    else:
      genes.append(gene)
      
  return genes

def crossover(individuals):
    new_individuals = []

    new_individuals.append(individuals[0])
    new_individuals.append(individuals[1])

    for i in range(2, no_of_individuals):
        new_individual = []
        if(i < (no_of_individuals - 2)):
            if(i == 2):
                parentA = random.choice(individuals[:3])
                parentB = random.choice(individuals[:3])
            else:
                parentA = random.choice(individuals[:])
                parentB = random.choice(individuals[:])

            for i in range(len(parentA)):
                n = random.random()
                if(n< 0.5):
                    new_individual.append(parentA[i])
                else:
                    new_individual.append(parentB[i])
         
        else:
            new_individual = random.choice(individuals[:])

        new_individuals.append(mutate(new_individual))
        #new_individuals.append(new_individual)

    return new_individuals

def evolve(individuals, fitness):
  
    sorted_y_idx_list = sorted(range(len(fitness)),key=lambda x:fitness[x])
    individuals = [individuals[i] for i in sorted_y_idx_list ]

    individuals.reverse()

    new_individuals = crossover(individuals)

    return new_individuals
