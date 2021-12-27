import random
import matplotlib.pyplot as plt
import time

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
        self.memory=[sum(distribution)]
    
    def simulate(self,toDie,toBirth):
        if toDie == True and toBirth == False:
            self.distribution.pop()
        elif toDie == False and toBirth == True:
            self.distribution.append(True)
        self.memory.append(sum(self.distribution))
        
    def plot(self):
        """To DO"""
        
        plt.plot(self.memory, label="Population of A")

        plt.ylabel('Number of A individuals')
        plt.xlabel('steps')
        plt.ylim([0, self.size+self.size*0.1]) #added 10% to the upper limit to help the vizualisation
        plt.legend()
        plt.show()

def calculate_Next_step(population):
    if not population.negative_selection:
        birththreshold=(population.memory[-1]*population.fitness)/((population.memory[-1]*population.fitness)+population.size-population.memory[-1])
    else:
        birththreshold=(population.memory[-1])/((population.memory[-1])+(population.size-population.memory[-1])*population.fitness)
    deaththreshold=population.memory[-1]/population.size
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

size=400
fitness=2
initial_distribution=[True]*(size//2)

popu=Population(size,initial_distribution,fitness,negative_selection=False)
step=40000

start=time.time()
for i in range(step):
    toBirth,toDie=calculate_Next_step(popu)
    popu.simulate(toDie,toBirth)

stop=time.time()
print(f"the simulation took {stop-start:.4f} seconds")
popu.plot()
