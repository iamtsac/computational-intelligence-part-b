import random
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from deap import creator, base, algorithms,tools

N=10000
df = pd.read_csv('data/mnist_train.csv')
images = df.loc[:, df.columns != 'label'].to_numpy().reshape(df.shape[0],28,28) 
images = tf.keras.utils.normalize(images)
images = images[:N]
#print(images[0])

labels = df['label'].to_numpy() 
default_labels= labels[:N]
labels = tf.keras.utils.to_categorical(labels,10)
labels = labels[:N]

model = tf.keras.models.load_model('model.h5') 

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 784)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness_func(individual):
    test = list()
    individual = np.array(individual).reshape(28,28)
    inputs = np.multiply(individual,images)
    loss, acc = model.evaluate(inputs,labels,verbose=0)
    predicts = model.predict(inputs) 
    classes = np.argmax(predicts, axis=1) 
    test.append(float((classes != default_labels).sum()/float(classes.size)))

    return test

toolbox.register("evaluate", fitness_func)

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

toolbox.register("select", tools.selTournament, tournsize=3)

#def main():
#
#    random.seed(64)
#    
#    pop = toolbox.population(n=20)
#    
#    # np equality function (operators.eq) between two arrays returns the
#    # equality element wise, which raises an exception in the if similar()
#    # check of the hall of fame. Using a different equality function like
#    # np.array_equal or np.allclose solve this issue.
#    hof = tools.HallOfFame(1)
#    
#    stats = tools.Statistics(lambda ind: ind.fitness.values)
#    stats.register("avg", np.mean)
#    stats.register("std", np.std)
#    stats.register("min", np.min)
#    stats.register("max", np.max)
#    
#    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.1, ngen=20, stats=stats,
#                        halloffame=hof)
#
#    return pop, stats, hof

def evolve(pop_size,crossover_pb,mutate_pb):
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=pop_size)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = crossover_pb, mutate_pb
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    check_criteria =0
    prev_fit = max(fits)
    # Begin the evolution
    while g < 1000 and check_criteria < 5:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        if max(fits) < (1.001*prev_fit) or prev_fit == max(fits):
            check_criteria += 1
        else:
            check_criteria = 0

        prev_fit = max(fits) 
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

evolve(20,0.6,0.0)
