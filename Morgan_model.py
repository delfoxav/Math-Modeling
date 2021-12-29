import random
import matplotlib.pyplot as plt
import time
import os
import csv
#import sys, getopt

from numpy import NaN, negative

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

def AbsorbingStateCalculation(size,fitness,initial_distribution,negative_selection):
    "Calculate the most probable Absorbing State with infinit number of steps"
    if fitness!=1:
        if negative_selection == False:
            #Equation 6.17
            try:
                p0 = (1-1/fitness**sum(initial_distribution))/(1-1/fitness**0)
            except ZeroDivisionError:
                p0=0
            try:
                pN = (1-1/fitness**sum(initial_distribution))/(1-1/fitness**size)
            except ZeroDivisionError:
                pN=0
        else:
            #Adaptation of equation 6.18
            try:
                pN = (1-1/fitness**(size-sum(initial_distribution)))/(1-1/fitness**0)
            except ZeroDivisionError:
                pN=0
            try:
                p0 = (1-1/fitness**(size-sum(initial_distribution)))/(1-1/fitness**size)
            except ZeroDivisionError:
                p0=0
       
    else:
        #Equation 6.5
        p0=(size-sum(initial_distribution))/size
        pN=sum(initial_distribution)/size
    if p0>pN:
        absorbingState="Full B"
    else:
        absorbingState= "Full A"
    
    return max(p0,pN), absorbingState





############Simulation#######################
def verification(step,size,fitness,initial_distribution,nbr_runs,negative_selection,output_file):
    """function to verify the implementation of the Moran model. Runs the model multiple time with the same initial parameters
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
    AbsorbtionAtA=0
    AbsorbtionAtB=0

    results=[]
    if negative_selection==True:
        fitnessOn='B'
    else:
        fitnessOn='A'

    for i in range(nbr_runs):
        popu=Population(size,initial_distribution.copy(),fitness,negative_selection=negative_selection)
        

        start=time.time()
        for j in range(step):
            toBirth,toDie=calculate_Next_step(popu)
            popu.simulate(toDie,toBirth)

        stop=time.time()
        print(f"The simulation took {stop-start:.4f} seconds, Simulation {i}/{nbr_runs}")
        #Store absorbing state:
        if popu.memory[-1] ==size:
            AbsorbtionAtA+=1
        elif popu.memory[-1] == 0:
            AbsorbtionAtB+=1
            
        #popu.plot()
        results.append(popu.memory)


    ###Plotting
    for i in range(len(results)):
        plt.plot(results[i], label="Run "+str(i+1))

    
    if nbr_runs <=10: #Avoid having plot with too many legends
        plt.legend()
    #plt.show()
    
    plt.savefig(output_file)
    plt.clf()
    if AbsorbtionAtA==AbsorbtionAtB and AbsorbtionAtA==0:
        mostProbableResultObserved="Didn't converge"
        observedProbability=NaN
    
    elif AbsorbtionAtA>=AbsorbtionAtB:
        mostProbableResultObserved="Full A"
        observedProbability=AbsorbtionAtA/(AbsorbtionAtA+AbsorbtionAtB)
    else:
        mostProbableResultObserved="Full B"
        observedProbability=AbsorbtionAtB/(AbsorbtionAtA+AbsorbtionAtB)
    CalculatedProbability,mostProbableResultCalculated= AbsorbingStateCalculation(size=size,fitness=fitness,initial_distribution=initial_distribution,negative_selection=negative_selection)
    
    headers=["Population size",
             "Initial number of A",
             "Initial number of B",
             "fitness on",
             "fitness",
             "number of Steps",
             "number of runs",
             "most probable result calculated",
             "calculated probability",
             "most probable result observed",
             "observed probability",]
    results=[{'Population size':str(size),
             'Initial number of A':str(sum(initial_distribution)),
             'Initial number of B':str(size-sum(initial_distribution)),
             'fitness on':fitnessOn,
             'fitness':str(fitness),
             'number of Steps':str(step),
             'number of runs':str(nbr_runs),
             'most probable result calculated':mostProbableResultCalculated,
             "calculated probability":CalculatedProbability,
             "most probable result observed":mostProbableResultObserved,
             "observed probability":observedProbability}]
    if not os.path.isfile("verification/result_verification.csv"): #Create the csv
        with open("verification/result_verification.csv",'w',encoding='UTF8') as f:
            writer = csv.DictWriter(f,fieldnames=headers)
            writer.writeheader()
            f.close()
    
    with open("verification/result_verification.csv","a",encoding="UTF8") as f:
        writer = csv.DictWriter(f,fieldnames=headers)
        writer.writerows(results)
        f.close()



################### Call the verification function from here ############################## 

size=20 #Size of the population
fitness=1.1 #selection coefficient
initial_distribution=[True]*(size//2) #initial distribution
nbr_runs=4000 #Number of runs (for the validation)
step=20000
negative_selection=True
output_file="verification/Test1.png"
verification(step=step,size=size,fitness=fitness,initial_distribution=initial_distribution,nbr_runs=nbr_runs,negative_selection=negative_selection,output_file=output_file)

