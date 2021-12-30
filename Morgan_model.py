import random
import matplotlib.pyplot as plt
import time
import os
import csv
# import sys, getopt
from typing import Union


from numpy import NaN, negative, mean


# TO DO WHAT IS THE LINK BETWEEN FITNESS AND SELECTION
## Nicolas: A higher fitness leads to a higher probability of fixation


class Population:
    """
    Population of two types of individual A and B

    :param int size:                total size of the population
    :param list[True] distribution: list of boolean representing the population
                                    The individuals A are represented by True and the individual B by nothing.
                                    Therefore, the birth and death all have the same computational cost
    :param float fitness:           selection on A or B
    :param bool negative_selection: if True the fitness favors B default False
    """

    def __init__(self, size: int, distribution: list[True], fitness: float = 1, negative_selection: bool = False):
        self.size = size
        self.fitness = fitness
        self.distribution = distribution
        self.negative_selection = negative_selection
        self.memory = [sum(self.distribution)]
        # DEAL WITH NEGATIVE SELECTION
        self.selection = self.fitness
        self.birthrates = [
            CalculateBirthRate(A=sum(self.distribution), B=size - sum(self.distribution), fitness=self.selection)]
        self.deathrates = [
            CalculateDeathRate(A=sum(self.distribution), B=size - sum(self.distribution), fitness=self.selection)]

    def simulate(self, toDie: bool, toBirth: bool) -> None:
        """
        Simulates a step of the Moran process
        Calculate step specific death and birth rate

        :param bool toDie:      individual for death True for A individual False for B individual
        :param bool toBirth:    individual for reproduction True for A individual False for B individual
        """
        if toDie == True and toBirth == False:
            self.distribution.pop()
        elif toDie == False and toBirth == True:
            self.distribution.append(True)
        self.memory.append(sum(self.distribution))
        self.deathrates.append(
            CalculateDeathRate(A=sum(self.distribution), B=self.size - sum(self.distribution), fitness=self.selection))
        self.birthrates.append(
            CalculateBirthRate(A=sum(self.distribution), B=self.size - sum(self.distribution), fitness=self.selection))

    def plot(self) -> None:
        """plot the number of A individuals over time"""

        plt.plot(self.memory, label="Population of A")

        plt.ylabel('Number of A individuals')
        plt.xlabel('steps')
        plt.ylim([0, self.size * 1.1])  # added 10% to the upper limit to help the vizualisation
        plt.legend()
        plt.show()

    def plot_death_birth_rates(self) -> None:
        """plot for birth and death rates over time
                death rate = birth rate * fitness
        """
        plt.plot(self.deathrates, label="death Rates")
        plt.plot(self.birthrates, label="birth Rates")
        plt.ylabel('rates')
        plt.xlabel('steps')
        # plt.ylim([0, 1])
        plt.legend()
        plt.show()


def calculate_Next_step(population: Population) -> tuple[bool, bool]:
    """Return boolean for birth and death

    :param Population population:
    :return:
    """
    if not population.negative_selection:
        birththreshold = (population.memory[-1] * population.fitness) / (
                    (population.memory[-1] * population.fitness) + population.size - population.memory[-1])
    else:
        birththreshold = (population.memory[-1]) / (
                    (population.memory[-1]) + (population.size - population.memory[-1]) * population.fitness)
    deaththreshold = population.memory[-1] / population.size
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


def AbsorbingStateCalculation(size: int, fitness: float, initial_distribution: list[True],
                              negative_selection: bool) -> tuple[Union[float, int], str]:
    """Calculate the most probable Absorbing State with infinit number of steps

    :param int size:                            size of the population
    :param float fitness:                       selection coefficient
    :param list[True] initial_distribution:     initial distribution (A/B) of the population
    :param boolean negative_selection:          is the selection on B?
    """
    if fitness != 1:
        if negative_selection == False:
            # Equation 6.17
            try:
                p0 = (1 - 1 / fitness ** sum(initial_distribution)) / (1 - 1 / fitness ** 0)
            except ZeroDivisionError:
                p0 = 0
            try:
                pN = (1 - 1 / fitness ** sum(initial_distribution)) / (1 - 1 / fitness ** size)
            except ZeroDivisionError:
                pN = 0
        else:
            # Adaptation of equation 6.18
            try:
                pN = (1 - 1 / fitness ** (size - sum(initial_distribution))) / (1 - 1 / fitness ** 0)
            except ZeroDivisionError:
                pN = 0
            try:
                p0 = (1 - 1 / fitness ** (size - sum(initial_distribution))) / (1 - 1 / fitness ** size)
            except ZeroDivisionError:
                p0 = 0

    else:
        # Equation 6.5
        p0 = (size - sum(initial_distribution)) / size
        pN = sum(initial_distribution) / size
    if p0 > pN:
        absorbingState = "Full B"
    else:
        absorbingState = "Full A"

    return max(p0, pN), absorbingState


def CalculateBirthRate(A: int, B: int, fitness: float) -> Union[float, int]:
    """Calculation of the birth rate of A in a population of A and B individuals with a selection coefficient between 0 and 1 in favor of A when fitness>1
    Without any mutation.

    :param int A:           number of A individuals (sum of distribution)
    :param int B:           number of B individuals (N - i; size - sum of distribution)
    :param float fitness:   selection coefficient
    :return:                rate of birth
    """
    try:
        BirthRate = (B / (A + B)) * (fitness * A / (fitness * A + B))
    except ZeroDivisionError:
        BirthRate = 0
    return BirthRate


def CalculateDeathRate(A: int, B: int, fitness: float) -> Union[float, int]:
    """Calculation of the death rate of A in a population of A and B individuals without any mutation.

    :param int A:           number of A individuals (sum of distribution)
    :param int B:           number of B individuals (N - i; size - sum of distribution)
    :param float fitness:   selection coefficient
    :return:                rate of death
    """
    try:
        DeathRate = (A / (A + B)) * (B / (fitness * A + B))
        # simplification of the DeathRate: fitness * Birthrate
    except ZeroDivisionError:
        DeathRate = 0
    return DeathRate


############Simulation#######################
def verification(step: int, size: int, fitness: float, initial_distribution: list[True], nbr_runs: int,
                 negative_selection: bool, output_file: str) -> None:
    """function to verify the implementation of the Moran model. Runs the model multiple time with the same initial parameters
    and plot the results

    :param int step:                            number of step for each simulation
    :param float fitness:                       selection coefficient
    :param int size:                            size of the population
    :param list[True] initial_distribution:     initial distribution (A/B) of the population
    :param int nbr_runs:                        number of runs for the verification
    :param boolean negative_selection:          is the selection on B?
    :param str output_file:                     name of the file to save the plot
    """
    AbsorbtionAtA = 0
    AbsorbtionAtB = 0

    results = []
    nbr_steps = []

    if negative_selection == True:
        fitnessOn = 'B'
    else:
        fitnessOn = 'A'

    for i in range(nbr_runs):
        popu = Population(size, initial_distribution.copy(), fitness, negative_selection=negative_selection)

        start = time.time()
        for j in range(step):
            toBirth, toDie = calculate_Next_step(popu)
            popu.simulate(toDie, toBirth)

            # stop simulation, when absorption state is reached
            if sum(popu.distribution) == 0 or sum(popu.distribution) == popu.size:
                nbr_steps.append(i)
                break

        stop = time.time()
        print(f"The simulation took {stop - start:.4f} seconds, Simulation {i}/{nbr_runs}")

        # Store absorbing state:
        if popu.memory[-1] == size:
            AbsorbtionAtA += 1
        elif popu.memory[-1] == 0:
            AbsorbtionAtB += 1

        # popu.plot()
        results.append(popu.memory)

    ###Plotting
    for i in range(len(results)):
        plt.plot(results[i], label="Run " + str(i + 1))

    if nbr_runs <= 10:  # Avoid having plot with too many legends
        plt.legend()
    # plt.show()

    plt.savefig(output_file)
    plt.clf()
    if AbsorbtionAtA + AbsorbtionAtB != nbr_runs:
        mostProbableResultObserved = "Didn't converge"
        observedProbability = NaN

    elif AbsorbtionAtA >= AbsorbtionAtB:
        mostProbableResultObserved = "Full A"
        observedProbability = AbsorbtionAtA / (AbsorbtionAtA + AbsorbtionAtB)
    else:
        mostProbableResultObserved = "Full B"
        observedProbability = AbsorbtionAtB / (AbsorbtionAtA + AbsorbtionAtB)
    CalculatedProbability, mostProbableResultCalculated = AbsorbingStateCalculation(size=size, fitness=fitness,
                                                                                    initial_distribution=initial_distribution,
                                                                                    negative_selection=negative_selection)

    if nbr_steps:
        max_nbr_steps = max(nbr_steps)
    else:
        max_nbr_steps = "Didn't converge"

    headers = ["Population size",
               "Initial number of A",
               "Initial number of B",
               "fitness on",
               "fitness",
               "number of Steps",
               "max number of Steps used",
               "number of runs",
               "most probable result calculated",
               "calculated probability",
               "most probable result observed",
               "observed probability", ]
    results = [{'Population size': str(size),
                'Initial number of A': str(sum(initial_distribution)),
                'Initial number of B': str(size - sum(initial_distribution)),
                'fitness on': fitnessOn,
                'fitness': str(fitness),
                'number of Steps': str(step),
                'max number of Steps used': max_nbr_steps,
                'number of runs': str(nbr_runs),
                'most probable result calculated': mostProbableResultCalculated,
                "calculated probability": CalculatedProbability,
                "most probable result observed": mostProbableResultObserved,
                "observed probability": observedProbability}]
    if not os.path.isfile("verification/result_verification.csv"):  # Create the csv
        with open("verification/result_verification.csv", 'w', encoding='UTF8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            f.close()

    with open("verification/result_verification.csv", "a", encoding="UTF8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerows(results)
        f.close()


################### Call the verification function from here ##############################

def main_verification():
    nbr_runs = 10000  # Number of runs (for the validation)
    step = 1000000
    negative_selection = False
    output_path = "verification"


    for size in [10, 100, 1000]:                    # sample size of 1000 might be to high
        for initdist in [1, size//2, size-1]:
            initial_distribution = [True] * initdist
            for fitness in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1, 1.01, 1.1, 2, 10, 100, 1000]:
                verification(step=step,size=size,fitness=fitness,initial_distribution=initial_distribution,
                             nbr_runs=nbr_runs,negative_selection=negative_selection,
                             output_file=f"{output_path}/size{size}_initdist{initdist}_fitness{fitness}.pdf")




#################### Birth Death Parameters ###################

def main_deathbirth():
    size = 100  # Size of the population
    fitness = 2  # selection coefficient
    initial_distribution = [True] * 50 # initial distribution
    step = 1000000

    popu = Population(size=size, distribution=initial_distribution, fitness=fitness)
    for j in range(step):
        toBirth, toDie = calculate_Next_step(popu)
        popu.simulate(toDie, toBirth)

        # stop simulation, when absorption state is reached
        if sum(popu.distribution) == 0 or sum(popu.distribution) == popu.size:
            break

    popu.plot_death_birth_rates()

if __name__ == "__main__":
    main_deathbirth()
