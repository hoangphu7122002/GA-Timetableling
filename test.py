import numpy as np

pop_and_child = np.array([8,2,3,4,5,6,7,1])
n_largest_index = pop_and_child.argsort()[-4:]
population = pop_and_child[n_largest_index]
print(population)