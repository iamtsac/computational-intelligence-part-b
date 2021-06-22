import random
import tensorflow as tf 
import numpy as np, numpy
import pandas as pd 
import matplotlib.pyplot as plt
from deap import creator, base, algorithms,tools

cases = [[20,0.6,0.00],
         [20,0.6,0.01],
         [20,0.6,0.10],
         [20,0.9,0.01],
         [20,0.1,0.01],
         [200,0.6,0.00],
         [200,0.6,0.01],
         [200,0.6,0.10],
         [200,0.9,0.01],
         [200,0.1,0.01]]

N=10000
df = pd.read_csv('data/mnist_train.csv')
images = df.loc[:, df.columns != 'label'].to_numpy().reshape(df.shape[0],28,28) 
images = tf.keras.utils.normalize(images)
images = images[:N]

df_test = pd.read_csv('data/mnist_test.csv')
images_test = df_test.loc[:, df_test.columns != 'label'].to_numpy().reshape(df_test.shape[0],28,28) 
images_test = tf.keras.utils.normalize(images_test)

labels_test = df_test['label'].to_numpy() 
labels_test = tf.keras.utils.to_categorical(labels_test,10)

labels = df['label'].to_numpy() 
default_labels= labels[:N]
labels = tf.keras.utils.to_categorical(labels,10)
labels = labels[:N]

model = tf.keras.models.load_model('model.h5') 

def fitness_func(individual):
    output = list()
    individual = np.array(individual).reshape(28,28)
    inputs = np.multiply(individual,images)
    loss, acc = model.evaluate(inputs,labels,verbose=0)
    ones = np.count_nonzero(individual==1)
    if ones > 300:
        output.append((acc - (loss/100)*(ones - 300)))
    else:
        output.append(acc)
    return output

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 784)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", fitness_func)

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

toolbox.register("select", tools.selTournament, tournsize=3)


def evolve(pop_size,crossover_pb,mutate_pb):
    random.seed(64) 

    logbook = tools.Logbook()
    # create an initial population 
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

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    g = 0 # Variable keeping track of the number of generations
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
        
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        

        if max(fits) < (1.001*prev_fit) or prev_fit == max(fits):
            check_criteria += 1
        else:
            check_criteria = 0

        prev_fit = max(fits) 
        
        
        logbook.header = ["gen"] + stats.fields
        print("  Max %s" % max(fits))
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", numpy.max)

        logbook.record(gen=g, **stats.compile(pop))
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    return logbook,best_ind

def for_plots():
    means = list()
    for case in cases:
        log = list() 
        df = pd.DataFrame() 
    
        print("CASE",case[0],case[1],case[2])
        for _ in range(2):
            logs,_ = evolve(case[0],case[1],case[2])
            log.append(logs)
        
        
        
        
        for i in range(0,2):
            df = pd.concat([df,pd.DataFrame(log[i])])
        
        num_of_gens=df.nunique(axis=0)['gen'] 
        plt.plot([df['max'].loc[(df['gen'] == x)].mean() for x in range(1,num_of_gens+1)]) 
        plt.xlabel("Generations")
        plt.ylabel("Best solution fitness")
        plt.savefig(str(case[0])+'_'+str(case[1])+'_'+str(case[2])+'.png')
        plt.clf()
        means.append([df['max'].mean(),df['gen'].mean()])
        
    
    
    with open('listfile.txt', 'w') as filehandle:
        for listitem in means:
            filehandle.write('%s\n' % listitem)
    

def best_sol(case=cases[8]):
    _,best = evolve(case[0],case[1],case[2])

    with open('best.txt', 'w') as filehandle:
        for listitem in best:
            filehandle.write('%s\n' % listitem)
    plt.imshow(np.multiply(np.array(best).reshape(28,28),images[0]),cmap='gray')
    plt.imshow(np.array(best).reshape(28,28),cmap='gray')
    plt.show()
    inputs = np.multiply(np.array(best).reshape(28,28),images)
    loss, acc = model.evaluate(inputs,labels,verbose=1)

def eval_best():
    best_solution = list()

    with open('best_solution.txt','r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]

            best_solution.append(currentPlace)
    best_solution = np.array(list(map(int,best_solution))).reshape(28,28)
    new_in = np.multiply(best_solution,images_test)
    print("model")
    model.evaluate(images_test,labels_test,verbose=1)
    print("ga")
    model.evaluate(new_in,labels_test,verbose=1)

#for_plots() #10 times for each case
best_sol() #best case 
eval_best() #check acc of best
