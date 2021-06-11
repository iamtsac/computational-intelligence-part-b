import random
import tensorflow as tf 
import numpy as np, numpy
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


def evolve(pop_size,crossover_pb,mutate_pb):
    random.seed(64) 

    logbook = tools.Logbook()
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=pop_size)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = crossover_pb, mutate_pb
    hof = tools.HallOfFame(1)
    
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

    stats = tools.Statistics(key=lambda ind: ind.fitness.values) 
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
        
        # Gather all the fitnesses in one list and print the stats
#        fits = [ind.fitness.values[0] for ind in offspring]
#        offspring = [x for _,x in sorted(zip(fits,offspring))]
        
        hof.update(pop)
#        pop[0] = hof
#        pop[1:] = offspring[:pop_size-1]
        
        pop[:] = offspring
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
        
        logbook.header = ["gen"] + stats.fields
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", numpy.max)

        logbook.record(gen=g, **stats.compile(pop))
    
    print("-- End of (successful) evolution --")
    
    #best_ind = tools.selBest(pop, 1)[0]
    #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    return logbook

log = list() 

for _ in range(2):
    log.append(evolve(20,0.6,0.0))



df = pd.DataFrame() 

for i in range(0,2):
    df = pd.concat([df,pd.DataFrame(log[i])])

num_of_gens=df.nunique(axis=0)['gen'] 
plt.plot([df['max'].loc[(df['gen'] == x)].mean() for x in range(1,num_of_gens+1)]) 
plt.show()


