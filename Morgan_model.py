import random
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import numpy as np


class Population:
    """
    Population of two types of individual A and B
    arg size: total size of the population
    type size: int
    arg distribution: list of boolean representing the population. The individuals A are represented by True and the individual B by nothing.
    Therefore, the birth and death all have the same computational cost
    type distribution: list(True)
    arg fitness: selection on A or B
    type fitness: double
    arg negative_selection: if True the fitness favors B default False
    type negative_selection: boolean
    """

    def __init__(self, size, distribution, fitness=1, negative_selection=False):
        self.size = size
        self.fitness = fitness
        self.distribution = distribution
        self.negative_selection = negative_selection
        self.memory = sum(distribution)

    def simulate(self, toDie, toBirth):
        if toDie == True and toBirth == False:
            self.distribution.pop()
        elif toDie == False and toBirth == True:
            self.distribution.append(True)
        self.memory = sum(self.distribution)

def calculate_Next_step(population):
    if not population.negative_selection:
        birththreshold = (population.memory * population.fitness) / (
                    (population.memory * population.fitness) + population.size - population.memory)
    else:
        birththreshold = (population.memory) / (
                    (population.memory) + (population.size - population.memory) * population.fitness)
    deaththreshold = population.memory / population.size
    birth = random.uniform(0, 1)
    death = random.uniform(0, 1)
    if birth <= birththreshold:
        birth = True
    else:
        birth = False
    if death <= deaththreshold:
        death = True
    else:
        death = False

    return (birth, death)


############Simulation#######################

def simulation(step, size, fitness, initial_distribution, output_path, negative_selection=False):
    """live simulation of the moran process
    all 10 steps the plot is saved as pdf file into the output_path

    param: step: number of step for each simulation (has to be square number for this plot style)
    param: fitness: selection coefficient
    param: size: size of the population
    param: initial_distribution: initial distribution (A/B) of the population
    param: negative_selection: is the selection on B?
    param: output_path: name of the path to save the plots

    type: step: int
    type: fitness: unsigned float
    type: size: int
    type initial_distribution: list of TRUE
    type negative_selection: boolean
    type output_path: str
    """
    popu = Population(size, initial_distribution, fitness, negative_selection=negative_selection)

    # Visualization
    fig, ax = plt.subplots(figsize=[5, 5])
    cmap = colors.ListedColormap([[152 / 255, 64 / 255, 99 / 255]])
    cmap.set_bad(color=[65 / 255, 67 / 255, 106 / 255])


    Z = initial_distribution + [False] * (size - len(initial_distribution))
    np.random.shuffle(Z)

    length = int(np.sqrt(size))  # only if size is square number
    Z = np.reshape(np.array(Z), (length, length))

    start = time.time()
    for i in range(step):
        toBirth, toDie = calculate_Next_step(popu)
        popu.simulate(toDie, toBirth)

        # Visualization
        Y = np.ma.masked_where(Z == False, Z)
        ax.cla()
        ax.imshow(Y, cmap=cmap, origin="lower")
        ax.set_title(f"step {i}                                      i/N = {np.sum(Z)}/{Z.size}")

        if i % 10 == 0:
            plt.savefig(f"{output_path}/{i}.pdf")
        plt.pause(0.01)

        if np.sum(Z) == Z.size or np.sum(Z) == 0:
            plt.savefig(f"{output_path}/{i}.pdf")
            plt.pause(3)
            break

        if toDie == True and toBirth == False:
            x, y = np.where(Z == True)
            if len(x) > 0:
                i = np.random.randint(len(x))
                random_pos = (x[i], y[i])
                Z[random_pos] = False

        if toDie == False and toBirth == True:
            x, y = np.where(Z == False)
            if len(x) > 0:
                i = np.random.randint(len(x))
                random_pos = (x[i], y[i])
                Z[random_pos] = True

    stop = time.time()
    print(f"the simulation took {stop - start:.4f} seconds")


size = 100
fitness = 2
initial_distribution = [True]
step = 100000
path = "visualization"

simulation(step, size, fitness, initial_distribution, path, negative_selection=False)
