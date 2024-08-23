from abc import abstractmethod, ABC
import numpy as np


class UCB:

    def __init__(self, numActions, ro=2):
        print("Using UCB-ro")
        self.numActions = numActions
        self.payoffMeans = [0] * self.numActions
        self.numPlays = [0] * self.numActions
        self.ro = ro
        self.t = 0

    def upperBound(self, arm_id: int):
        numPlays = self.numPlays[arm_id]
        step = self.t
        return self.payoffMeans[arm_id] + np.sqrt(self.ro * np.log(step + 1) / numPlays)

    def coldstart_id(self):
        if self.t < self.numActions:
            return self.t
    
    def suggest(self):
        coldstart = self.coldstart_id()
        if coldstart is not None:
            return coldstart

        ucbs = [self.upperBound(i)
                for i in range(self.numActions)]
        action = max(range(self.numActions), key=lambda i: ucbs[i])
        return action
    
    def update(self, action, reward):
        self.payoffMeans[action] = ((self.payoffMeans[action]*self.numPlays[action] + reward)/(self.numPlays[action]+1)).numpy()[0]
    
    def observe(self, action, reward):
        self.update(action, reward)
        self.numPlays[action] += 1
        self.t += 1
        
    def rep_stat(self):
        return f"Number of plays {self.numPlays}\nMean estimates {self.payoffMeans}"


class ISMUCB(UCB):
    
    def __init__(self, numActions):
        print("Using ISMUCB")
        super().__init__(numActions, 2)
        self.payoffVars = [0] * self.numActions
        self.obs = {i: [] for i in range(numActions)}
    
    def coldstart_id(self):
        if self.t < 3*self.numActions:
            return self.t % self.numActions
        
    def update(self, action, reward):
        self.obs[action].append(reward.numpy()[0])
        super().update(action, reward)
        if self.numPlays[action] != 0:
            self.payoffVars[action] = np.sum((np.array(self.obs[action])-self.payoffMeans[action])**2)/(self.numPlays[action])
    
    def upperBound(self, arm_id: int):
        numPlays = self.numPlays[arm_id]
        step = self.t
        return self.payoffMeans[arm_id] + np.sqrt(self.payoffVars[arm_id]*(self.t**(2/(self.numPlays[arm_id]-2))-1))
        
    def rep_stat(self):
        sup_st = super().rep_stat() 
        return sup_st+"\n"+f"Var stat {self.payoffVars}" 
    

if __name__ == "__main__":
    ucb = UCB(5)
    for i in range(1000):
        a = ucb.suggest()
        ucb.observe(a, a/10+np.sqrt(a+1)*np.random.randn())
        print(a, end=", ")
    print(ucb.rep_stat())
