import ga
import random
import numpy as np
import pandas as pd
import statistics
import sys

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 40
num_parents_mating = 16
# Creating the initial population.
population = ga.createParent(sol_per_pop)
pop_size = population.shape

best_outputs = []
num_generations = 100
mutation_rate = 0.3
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = ga.cal_pop_fitness(population)
    print("Fitness")
    print(fitness)

    best_outputs.append(np.max(fitness))
    # The best result in the current iteration.
    print("Best result : ", np.max(fitness))

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(population,
                                    num_parents_mating)
    # print("Parents")
    # print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,
                                       offspring_size= parents.shape[0])
    # print("Crossover")
    # print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, mutation_rate)
    # print("Mutation")
    # Creating the new population based on the parents and offspring.
    pop_and_child = np.concatenate((population,offspring_mutation))
    pop_and_child_fitness = ga.cal_pop_fitness(pop_and_child)
    #get n-largest element from pop_and_child
    n_largest_index = pop_and_child_fitness.argsort()[-pop_size[0]:]
    population = pop_and_child[n_largest_index]
    # new_population[0:parents.shape[0], :] = parents
    # new_population[parents.shape[0]:, :] = offspring_mutation

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.argmax(fitness == np.max(fitness))
best_result = population[best_match_idx, :]
for chromosome_index in range(best_result.shape[0]):
    chromosome = best_result[chromosome_index]
    chromosome = str(chromosome).split('-')
    chromosome[-1] = ga.decode_datetime(chromosome[-1])
    best_result[chromosome_index] = '-'.join(chromosome)
# print("Best solution : ", best_result)
print("Best solution fitness : ", fitness[best_match_idx])

import matplotlib.pyplot

matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
# Save result to out.csv
import pandas as pd

df = pd.DataFrame(best_result)
df = df[0].str.split('-', expand=True)
df.columns = ['wonum', 'targstartdate', 'targcompdate', 'schedstartdate']
from pathlib import Path

filepath = Path('out.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath)
