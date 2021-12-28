import random
import matplotlib.pyplot as plt
import time

from numpy import negative

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
def verification(step,size,fitness,initial_distribution,nbr_runs,negative_selection,output_file):
    """fonction to verify the implementation of the Moran model. Runs the model multiple time with the same initial parameters
    and plot the results
    param: step: number of step for each simulation
    param: fitness: selection coefficient
    param: size: size of the population
    param: initial_distribution: initial distribution (A/B) of the population
    param: nbr_runs: number of runs for the verification
    param: negative_selection: is the selection on B? 
    param: output_file: name of the file to save the plot
    
    type: step: int
    type: fitness: unsigned float
    type: size: int
    type initial_distribution: list of TRUE
    type nbr_runs: int
    type negative_selection: boolean
    type output_file: str
    """

    results=[]

    for i in range(nbr_runs):
        popu=Population(size,initial_distribution.copy(),fitness,negative_selection=negative_selection)
        

        start=time.time()
        for i in range(step):
            toBirth,toDie=calculate_Next_step(popu)
            popu.simulate(toDie,toBirth)

        stop=time.time()
        print(f"the simulation took {stop-start:.4f} seconds")
        #popu.plot()
        results.append(popu.memory)


    ###Plotting
    for i in range(len(results)):
        plt.plot(results[i], label="Run "+str(i+1))

    plt.legend()
    #plt.show()
    plt.savefig(output_file)
    plt.clf()



################### Call the verification function from here ############################## 

size=20 #Size of the population
fitness=2 #selection coefficient
initial_distribution=[True]*(size//2) #initial distribution
nbr_runs=4 #Number of runs (for the validation)
step=10000
negative_selection=False
output_file="verification/Test1.png"
verification(step=step,size=size,fitness=fitness,initial_distribution=initial_distribution,nbr_runs=nbr_runs,negative_selection=negative_selection,output_file=output_file)

size=400 #Size of the population
fitness=2 #selection coefficient
initial_distribution=[True]*(size//2) #initial distribution
step=200
nbr_runs=4 #Number of runs (for the validation)
negative_selection=False
output_file="verification/Test2.png"

verification(step=step,size=size,fitness=fitness,initial_distribution=initial_distribution,nbr_runs=nbr_runs,negative_selection=negative_selection,output_file=output_file)


size=40000 #Size of the population
fitness=2 #selection coefficient
initial_distribution=[True]*(size//2) #initial distribution
step=200
nbr_runs=4 #Number of runs (for the validation)
negative_selection=False
output_file="verification/Test3.png"

verification(step=step,size=size,fitness=fitness,initial_distribution=initial_distribution,nbr_runs=nbr_runs,negative_selection=negative_selection,output_file=output_file)
