import random
import matplotlib.pyplot as plt
import time

class Population:
    #uses numpy array
    def __init__(self,size,distribution,fitness=1, negative_selection=False):
        self.size=size
        self.fitness=fitness
        self.distribution=distribution
        self.negative_selection=negative_selection
        self.memory=[sum(distribution)]
    
    def death(self,type):
        """Death of an individual"""
        if type==True:
            self.distribution.pop()
        elif type == False:
            self.distribution.append(True)
        
    def birth(self,type):
        """Birth of a new individual"""
        if type== True:
            self.distribution.append(True)
        elif type == False:
            self.distribution.pop()
    
    
    def simulate(self,toDie,toBirth):
        self.birth(toBirth)
        self.death(toDie)
        self.memory.append(sum(self.distribution))
        
    def plot(self):
        plt.plot(self.memory)
        plt.ylabel('Number of A individuals')
        plt.xlabel('steps')
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

size=40000

popu=Population(size,[True]*(size//2),2,negative_selection=True)
step=40000


start=time.time()
for i in range(step):
    toBirth,toDie=calculate_Next_step(popu)
    popu.simulate(toDie,toBirth)

stop=time.time()
print(stop-start)
popu.plot()