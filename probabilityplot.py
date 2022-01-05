import matplotlib.pyplot as plt
import numpy as np

def pfunction(N: int, i: int, r: float) -> float:
    """
    calculates the probability of fixation according to equation (5) and (9)

    :param int N:   population size
    :param int i:   number of initial individuals A
    :param float r: fitness of A
    :return:        probability of fixation
    """
    if r == 1:
        return i/N
    if (1-1/r**N) == 0:
        p = 0.0
    else:
        p = (1-1/r**i)/(1-1/r**N)
    return p


# Plot i = 1
plt.style.use("seaborn-talk")
plt.subplots(figsize=[10, 10])
X = np.arange(1, 100, 0.1)
Y = {}
# different fitness values
for r in [0.5, 0.9, 1, 1.1, 2]:
        Y[r] = [pfunction(N, 1, r) for N in X]
        plt.plot(X, Y[r], label=f"i = 1; r = {r}")

plt.legend(frameon=False)

plt.plot()
plt.ylabel('fixation probability', labelpad=15)
plt.xlabel('population size (N)', labelpad=15)

plt.show()


# Plot i = N//2
plt.subplots(figsize=[10, 10])
X = np.arange(1, 100, 0.1)
Y = {}
# different fitness values
for r in [0.5, 0.9, 1, 1.1, 2]:
    Y[r] = [pfunction(N, N // 2, r) for N in X]
    plt.plot(X, Y[r], label=f"i = \u230AN/2\u230B; r = {r}")

plt.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 0.9))

plt.plot()
plt.ylabel('fixation probability', labelpad=15)
plt.xlabel('population size (N)', labelpad=15)

plt.show()


# Plot i = N-1
plt.subplots(figsize=[10, 10])
X = np.arange(1, 100, 0.1)
Y = {}
# different fitness values
for r in [0.5, 0.9, 1, 1.1, 2]:
    Y[r] = [pfunction(N, N - 1, r) for N in X]
    plt.plot(X, Y[r], label=f"i = N-1; r = {r}")

plt.legend(frameon=False)

plt.plot()
plt.ylabel('fixation probability', labelpad=15)
plt.xlabel('population size (N)', labelpad=15)

plt.show()
