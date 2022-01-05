import random
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import os
import csv
from typing import Union
from numpy import NaN, random, reshape, sqrt, array, ma, where, logical_not, sum as npsum


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
        self.selection = self.fitness
        self.birthrates = [
            CalculateBirthRate(A=sum(self.distribution), B=size - sum(self.distribution), fitness=self.selection,
                               selectionOnB=negative_selection)]
        self.deathrates = [
            CalculateDeathRate(A=sum(self.distribution), B=size - sum(self.distribution), fitness=self.selection,
                               selectionOnB=negative_selection)]

    def simulate(self, toDie: bool, toBirth: bool) -> None:
        """
        Simulates a step of the Moran process
        Calculate step specific death and birth rate

        :param bool toDie:      individual for death True for A individual False for B individual
        :param bool toBirth:    individual for reproduction True for A individual False for B individual
        """
        if toDie and not toBirth:
            self.distribution.pop()
        elif not toDie and toBirth:
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
        plt.legend(frameon=False)
        plt.show()

    def plot_death_birth_rates(self) -> None:
        """plot for birth and death rates over time
                death rate = birth rate * fitness
        """
        plt.plot(self.deathrates, label="death Rates")
        plt.plot(self.birthrates, label="birth Rates")
        plt.ylabel('rates')
        plt.xlabel('steps')
        plt.legend(frameon=False)
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

    return birth, death


def AbsorbingStateCalculation(size: int, fitness: float, initial_distribution: list[True],
                              negative_selection: bool) -> tuple[Union[float, int], str]:
    """Calculate the most probable Absorbing State with infinit number of steps

    :param int size:                            size of the population
    :param float fitness:                       selection coefficient
    :param list[True] initial_distribution:     initial distribution (A/B) of the population
    :param boolean negative_selection:          is the selection on B?
    """
    if fitness != 1:
        if not negative_selection:
            # Equation 6.17
            try:
                pN = (1 - 1 / fitness ** sum(initial_distribution)) / (1 - 1 / fitness ** size)
            except ZeroDivisionError:
                pN = 0
            p0 = 1 - pN
        else:
            # Adaptation of equation 6.18
            try:
                p0 = (1 - 1 / fitness ** (size - sum(initial_distribution))) / (1 - 1 / fitness ** size)
            except ZeroDivisionError:
                p0 = 0
            pN = 1 - p0

    else:
        # Equation 6.5
        p0 = (size - sum(initial_distribution)) / size
        pN = sum(initial_distribution) / size
    if p0 > pN:
        absorbingState = "Full B"
    else:
        absorbingState = "Full A"

    return max(p0, pN), absorbingState


def CalculateBirthRate(A: int, B: int, fitness: float, selectionOnB: bool = False) -> Union[float, int]:
    """Calculation of the birth rate of A in a population of A and B individuals
    with a selection coefficient between 0 and 1 in favor of A when fitness>1
    Without any mutation. If the selection is on B calculate the birth rate of B instead.

    :param int A:               number of A individuals (sum of distribution)
    :param int B:               number of B individuals (N - i; size - sum of distribution)
    :param float fitness:       selection coefficient
    :param bool selectionOnB:   is the selection on B?
    :return:                    rate of birth
    """
    if selectionOnB:
        A, B = B, A

    try:
        BirthRate = (B / (A + B)) * (fitness * A / (fitness * A + B))
    except ZeroDivisionError:
        BirthRate = 0
    return BirthRate


def CalculateDeathRate(A: int, B: int, fitness: float, selectionOnB: bool = False) -> Union[float, int]:
    """Calculation of the death rate of A in a population of A and B individuals without any mutation.
        If the selection is on B, calculate the death rate of B instead.

    :param int A:           number of A individuals (sum of distribution)
    :param int B:           number of B individuals (N - i; size - sum of distribution)
    :param float fitness:   selection coefficient
    :param selectionOnB:    is the selection on B?
    :return:                rate of death
    """
    if selectionOnB:
        A, B = B, A

    try:
        DeathRate = (A / (A + B)) * (B / (fitness * A + B))
        # simplification of the DeathRate: fitness * Birthrate
    except ZeroDivisionError:
        DeathRate = 0
    return DeathRate


def livesimulation(step: int, size: int, fitness: float, initial_distribution: list[True],
                   output_path: str, negative_selection: bool = False):
    """simulation with live images (plots) for each step
    :param int step:                            number of step for each simulation
    :param float fitness:                       selection coefficient
    :param int size:                            size of the population (for livesimulation square number required)
    :param list[True] initial_distribution:     initial distribution (A/B) of the population
    :param str output_path:                     name of the path to save every tenth plot
    :param boolean negative_selection:          is the selection on B?
    """
    if not os.path.isdir(output_path):  # create the directory to store every tenth plot
        os.mkdir(output_path)

    popu = Population(size, initial_distribution, fitness, negative_selection=negative_selection)

    # Visualization
    fig, ax = plt.subplots(figsize=[5, 5])
    cmap = colors.ListedColormap([[152/255, 64/255, 99/255]])
    cmap.set_bad(color=[65/255, 67/255, 106/255])

    Z = initial_distribution + [False] * (size - len(initial_distribution))
    random.shuffle(Z)
    length = int(sqrt(size))  # only if size is square number
    Z = reshape(array(Z), (length, length))

    start = time.time()
    for i in range(step):
        toBirth, toDie = calculate_Next_step(popu)
        popu.simulate(toDie=toDie, toBirth=toBirth)

        # Visualization
        Y = ma.masked_where(logical_not(Z), Z)
        ax.cla()
        ax.imshow(Y, cmap=cmap, origin="lower")
        ax.set_title(f"step {i}                                      i/N = {npsum(Z)}/{Z.size}")

        if i % 10 == 0:
            plt.savefig(f"{output_path}/{i}.pdf")
        plt.pause(0.01)

        if npsum(Z) == Z.size or npsum(Z) == 0:
            plt.savefig(f"{output_path}/{i}.pdf")
            plt.pause(3)
            break

        if toDie and not toBirth:
            x, y = where(Z)
            if len(x) > 0:
                i = random.randint(len(x))
                random_pos = (x[i], y[i])
                Z[random_pos] = False

        if not toDie and toBirth:
            x, y = where(logical_not(Z))
            if len(x) > 0:
                i = random.randint(len(x))
                random_pos = (x[i], y[i])
                Z[random_pos] = True

    stop = time.time()
    print(f"The simulation took {stop - start:.4f} seconds")


def verification(step: int, size: int, fitness: float, initial_distribution: list[True], nbr_runs: int,
                 negative_selection: bool, output_file: str) -> None:
    """function to verify the implementation of the Moran model.
    Runs the model multiple time with the same initial parameters and plot the results

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

    if not os.path.isdir("verification"):  # create the directory to store the verification files
        os.mkdir("verification")

    if negative_selection:
        fitnessOn = 'B'
    else:
        fitnessOn = 'A'

    for i in range(nbr_runs):
        popu = Population(size, initial_distribution.copy(), fitness, negative_selection=negative_selection)

        start = time.time()
        for j in range(step):
            toBirth, toDie = calculate_Next_step(popu)
            popu.simulate(toDie=toDie, toBirth=toBirth)

            # stop simulation, when absorption state is reached
            if sum(popu.distribution) == 0 or sum(popu.distribution) == popu.size:
                nbr_steps.append(j)
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

    # Plotting
    for i in range(len(results)):
        plt.plot(results[i], label="Run " + str(i + 1))

    if nbr_runs <= 10:  # Avoid having plot with too many legends
        plt.legend(frameon=False)

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
        
    CalculatedProbability, mostProbableResult = AbsorbingStateCalculation(size=size, fitness=fitness,
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
               "observed probability"]
    results = [{'Population size': str(size),
                'Initial number of A': str(sum(initial_distribution)),
                'Initial number of B': str(size - sum(initial_distribution)),
                'fitness on': fitnessOn,
                'fitness': str(fitness),
                'number of Steps': str(step),
                'max number of Steps used': max_nbr_steps,
                'number of runs': str(nbr_runs),
                'most probable result calculated': mostProbableResult,
                "calculated probability": CalculatedProbability,
                "most probable result observed": mostProbableResultObserved,
                "observed probability": observedProbability}]

    if not os.path.isfile("verification/result_verification.csv"):  # Create the csv
        with open("verification/result_verification.csv", 'w', encoding='UTF8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    with open("verification/result_verification.csv", "a", encoding="UTF8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerows(results)


def deathbirth(size: int, initial_distribution: list[True], fitness: float, step: int, output_file: str) -> None:
    """
    function to verify the impact of the initial parameters on the birth and death rates.
    Plot the evolution of the birth and death rate and store the value in a csv file

    :param int size:                            size of the population
    :param list[True] initial_distribution:     initial distribution (A/B) of the population
    :param float fitness:                       selection coefficient
    :param int step:                            number of step for each simulation
    :param str output_file:                     name of the file to save the plot
    """

    popu = Population(size=size, distribution=initial_distribution, fitness=fitness)

    headers = ["Step",
               "Birth rate",
               "Death rate", ]
    initdist = sum(initial_distribution)

    filename = f"size{size}_initdist{initdist}_fitness{fitness}.csv"

    if not os.path.isdir("death_birth"):  # create the directory to store the deathbirth files
        os.mkdir("death_birth")

    with open(f"death_birth/{filename}", 'w', encoding='UTF8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    for j in range(step):
        toBirth, toDie = calculate_Next_step(popu)
        popu.simulate(toDie=toDie, toBirth=toBirth)

        results = [{'Step': str(j),
                    'Birth rate': str(popu.birthrates[j]),
                    'Death rate': str(popu.deathrates[j])}
                   ]

        with open(f"death_birth/{filename}", "a", encoding="UTF8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerows(results)

        # stop simulation, when absorption state is reached
        if sum(popu.distribution) == 0 or sum(popu.distribution) == popu.size:
            break

    plt.style.use("seaborn-talk")
    fig, ax1 = plt.subplots(figsize=[12, 6])

    lns1 = ax1.plot(popu.deathrates, label="death rate", color="navy")
    lns2 = ax1.plot(popu.birthrates, label="birth rate", color="purple")
    ax1.tick_params(axis='y', labelcolor="rebeccapurple")
    ax1.set_ylabel('rate', color="rebeccapurple")

    ax2 = ax1.twinx()
    lns3 = ax2.plot(popu.memory, label="population of A", color="orangered")
    ax2.tick_params(axis='y', labelcolor="orangered")
    ax2.set_ylabel("population", color="orangered")

    # added these three lines
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", frameon=False)


    ax1.set_xlabel('steps')
    plt.savefig(f"death_birth/{output_file}")
    plt.show()
    plt.clf()


def report_livevisualization():
    """Live visualization"""
    step = 10000
    size = 100      # must be a square number
    fitness = 2
    initial_distribution = [True]
    output_path = "visualization"

    livesimulation(step=step, size=size, fitness=fitness, initial_distribution=initial_distribution,
                   output_path=output_path, negative_selection=False)


def report_verification():
    """Verification"""
    nbr_runs = 1000  # Number of runs (for the validation)
    step = 1000000
    negative_selection = False
    output_path = "verification"

    for size in [10, 100]:
        for initdist in [1, size // 2, size - 1]:
            initial_distribution = [True] * initdist
            for fitness in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1, 1.01, 1.1, 2, 10, 100, 1000]:
                verification(step=step, size=size, fitness=fitness, initial_distribution=initial_distribution,
                             nbr_runs=nbr_runs, negative_selection=negative_selection,
                             output_file=f"{output_path}/size{size}_initdist{initdist}_fitness{fitness}.pdf")


def report_deathbirth():
    """Birth Death Parameters"""
    deathbirth(size=100, initial_distribution=[True], fitness=2, step=100000, output_file="deathbirthplot.pdf")


# by uncommenting, the code used in the report can be run
if __name__ == "__main__":
    report_livevisualization()
    # report_verification()
    # report_deathbirth()
