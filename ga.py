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

def main():

    random.seed(64)
    
    pop = toolbox.population(n=20)
    
    # np equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # np.array_equal or np.allclose solve this issue.
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.1, ngen=20, stats=stats,
                        halloffame=hof)

    return pop, stats, hof

pop, stat, hof = main()

