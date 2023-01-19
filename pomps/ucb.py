from abc import abstractmethod, ABC

import numpy as np


class UCB:

    def __init__(self, numActions, ro=2):
        self.numActions = numActions
        self.payoffMeans = [0] * self.numActions
        self.numPlays = [0] * self.numActions
        self.ucbs = [0] * self.numActions
        self.ro = ro
        self.t = 0

    def upperBound(self, step, numPlays):
        return np.sqrt(self.ro * np.log(step + 1) / numPlays)

    def suggest(self):
        if self.t < self.numActions:
            return self.t

        ucbs = [self.payoffMeans[i] + self.upperBound(self.t, self.numPlays[i])
                for i in range(self.numActions)]
        action = max(range(self.numActions), key=lambda i: ucbs[i])
        return action

    def observe(self, action, reward):
        self.payoffMeans[action] = (self.payoffMeans[action]*self.numPlays[action] + reward)/(self.numPlays[action]+1)
        self.numPlays[action] += 1
        self.t += 1


if __name__ == "__main__":
    ucb = UCB(5)
    for i in range(10000):
        a = ucb.suggest()
        ucb.observe(a, a+10*np.random.randn())
    print('s')