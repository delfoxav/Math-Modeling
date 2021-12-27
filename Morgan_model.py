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
    
    
    def __init__(self,size,distribution,fitness=1, negative_selection=False):
        self.size=size
        self.fitness=fitness
        self.distribution=distribution
        self.negative_selection=negative_selection
        self.memory=sum(distribution)
    

    def simulate(self,toDie,toBirth):
        if toDie == True and toBirth == False:
            self.distribution.pop()
        elif toDie == False and toBirth == True:
            self.distribution.append(True)
        self.memory=sum(self.distribution)
        
    def plot(self):
        """To DO"""
        
        plt.plot(self.memory)
        plt.ylabel('Number of A individuals')
        plt.xlabel('steps')
        plt.show()

def calculate_Next_step(population):
    if not population.negative_selection:
        birththreshold=(population.memory*population.fitness)/((population.memory*population.fitness)+population.size-population.memory)
    else:
        birththreshold=(population.memory)/((population.memory)+(population.size-population.memory)*population.fitness)
    deaththreshold=population.memory/population.size
    birth=random.uniform(0,1)
    death=random.uniform(0,1)
    if birth <= birththreshold:
        birth=True
    else:
        birth = False
    if death <=deaththreshold:
        death=True
    else:
        death=False
    
    return(birth,death)




############Simulation#######################

size=100
length = int(np.sqrt(size)) # only if size is square number

fitness=2
initial_distribution=[True]*(size//2)

popu=Population(size,initial_distribution,fitness,negative_selection=False)
step=100000

# Visualization
fig, ax = plt.subplots()
cmap = colors.ListedColormap([[46/255, 139/255, 87/255]])
cmap.set_bad(color=[75/255, 0, 130/255])

Z = [True] * (size // 2) + [False] * (size - (size // 2))
np.random.shuffle(Z)
Z = np. reshape(np.array(Z), (length, length))

start=time.time()
for i in range(step):
    toBirth,toDie=calculate_Next_step(popu)
    popu.simulate(toDie,toBirth)

    # Visualization
    Y = np.ma.masked_where(Z == False, Z)
    ax.cla()
    ax.imshow(Y, cmap=cmap)
    ax.set_title(f"step {i}                                               {np.sum(Z)}/{Z.size}")
    plt.pause(0.01)

    if np.sum(Z) == Z.size or np.sum(Z) == 0:
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
    
stop=time.time()
print(f"the simulation took {stop-start:.4f} seconds")
