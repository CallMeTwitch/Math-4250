from typing import Union
import numpy as np

class Agent:
    '''
    Base class for all bandit agents.
    '''

    def __init__(self, numActions: int) -> None:
        '''
        Initialize agent.

        Keyword Arguments:
            numActions (int): Number of actions to select from.
        '''

        self.actionCounts = np.ones(numActions)
        self.QValues = np.zeros(numActions)
        self.numActions = numActions
        self.rewards = []
        self.actions = []

    def selectAction(self) -> int:
        '''
        Select action using agent strategy.

        Returns:
            int: Index of action selected.
        '''

        raise NotImplementedError('Method should be overridden by subclass.')
    
    def update(self, reward: float, action: int) -> None:
        '''
        Update internal state using action selected and reward received.

        Keyword Arguments:
            reward (float): Reward received from selecting action.
            action (int): Index of action selected.
        '''

        self.actionCounts[action] += 1
        self.QValues[action] += (reward - self.QValues[action]) / self.actionCounts[action]

        self.rewards.append(reward)
        self.actions.append(action)

class GreedyAgent(Agent):
    '''
    Greedy agent for the k-armed bandit problem.
    '''

    def __init__(self, numActions: int) -> None:
        '''
        Initialize greedy agent.

        Keyword Arguments:
            numActions (int): Number of actions to select from.
        '''
        
        super().__init__(numActions)

    def selectAction(self) -> int:
        '''
        Randomly select action with greatest Q value.

        Returns:
            int: Index of action selected.
        '''

        return np.random.choice(np.flatnonzero(self.QValues == self.QValues.max()))
    
    def update(self, reward: float, action: int) -> None:
        '''
        Update internal state using action selected and reward received.

        Keyword Arguments:
            reward (float): Reward received from selecting action.
            action (int): Index of action selected.
        '''

        super().update(reward, action)

class EpsilonGreedyAgent(Agent):
    '''
    Epsilon greedy agent for the k-armed bandit problem.
    '''

    __slots__ = 'epsilon', 'stepSize'
    def __init__(self, numActions: int, epsilon: float, stepSize: Union[float, None] = None) -> None:
        '''
        Initialize greedy agent.

        Keyword Arguments:
            numActions (int): Number of actions to select from.
            epsilon (float): Decimal percent of actions to be selected randomly.
            stepSize (Union[float, None]): Step size if used, None if incremental strategy is used.
        '''
        
        super().__init__(numActions)

        self.stepSize = stepSize
        self.epsilon = epsilon

    def selectAction(self) -> int:
        '''
        Randomly select action with greatest Q value with probability 1 - self.epsilon, randomly select action with probability self.epsilon.

        Returns:
            int: Index of action selected.
        '''

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.numActions)
        
        return np.random.choice(np.flatnonzero(self.QValues == self.QValues.max()))
    
    def update(self, reward: float, action: int) -> None:
        '''
        Update internal state using action selected and reward received.

        Keyword Arguments:
            reward (float): Reward received from selecting action.
            action (int): Index of action selected.
        '''

        if self.stepSize is None:
            super().update(reward, action)
        else:
            self.QValues[action] += self.stepSize * (reward - self.QValues[action])
            self.rewards.append(reward)
            self.actions.append(action)

class OptimisticAgent(Agent):
    '''
    Optimistic agent for the k-armed bandit problem.
    '''

    __slots__ = 'initValue',
    def __init__(self, numActions: int, initValue: float) -> None:
        '''
        Initialize greedy agent.

        Keyword Arguments:
            numActions (int): Number of actions to select from.
            initValue (float): Inital Q value.
        '''
        
        super().__init__(numActions)

        self.QValues = np.full(numActions, initValue, dtype = np.float32)
        self.initValue = initValue

    def selectAction(self) -> int:
        '''
        Randomly select action with greatest Q value.

        Returns:
            int: Index of action selected.
        '''

        return np.random.choice(np.flatnonzero(self.QValues == self.QValues.max()))
    
    def update(self, reward: float, action: int) -> None:
        '''
        Update internal state using action selected and reward received.

        Keyword Arguments:
            reward (float): Reward received from selecting action.
            action (int): Index of action selected.
        '''

        super().update(reward, action)

class GradientAgent(Agent):
    '''
    Gradient agent for the k-armed bandit problem.
    '''

    __slots__ = 'alpha',
    def __init__(self, numActions: int, alpha: float) -> None:
        '''
        Initialize greedy agent.

        Keyword Arguments:
            numActions (int): Number of actions to select from.
            alpha (float): Learning rate.
        '''
        
        super().__init__(numActions)

        self.H = np.zeros(numActions, dtype = np.float32)
        self.avgReward = 0.0
        self.timestep = 0
        self.alpha = alpha

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        '''
        Calculates numerically stable softmax.

        Keyword Arguments:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Softmax of input array.
        '''

        expX = np.exp(x - np.max(x))
        return expX / np.sum(expX)

    def selectAction(self) -> int:
        '''
        Randomly select action with probabilities learned.

        Returns:
            int: Index of action selected.
        '''

        return np.random.choice(self.numActions, p = self.softmax(self.H))
    
    def update(self, reward: float, action: int) -> None:
        '''
        Update internal state using action selected and reward received.

        Keyword Arguments:
            reward (float): Reward received from selecting action.
            action (int): Index of action selected.
        '''

        super().update(reward, action)

        self.timestep += 1
        self.avgReward += (reward - self.avgReward) / self.timestep

        mask = np.eye(self.numActions)[action]
        self.H += self.alpha * (reward - self.avgReward) * (mask - self.softmax(self.H))

class CustomAgent(Agent):
    '''
    Custom agent for the k-armed bandit problem.
    '''

    __slots__ = 'discount',
    def __init__(self, numActions: int, discount: Union[float] = 1.0) -> None:
        '''
        Initialize greedy agent.

        Keyword Arguments:
            numActions (int): Number of actions to select from.
            initValue (float): Inital Q value.
            discount (float): Weighting of value over previous value.
        '''
        
        super().__init__(numActions)

        self.sumsSquared = np.zeros(numActions, dtype = np.float32)
        self.sums = np.zeros(numActions, dtype = np.float32)
        self.wSum = np.full(numActions, discount, dtype = np.float32)

        self.xValues = np.linspace(-5, 5, 50)
        self.discount = discount

    @staticmethod
    def pdf(x: np.ndarray) -> np.ndarray:
        '''
        Estimate the normal probability density function.

        Keyword Arguments:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: An estimation of the probability of a normally distributed random variable occurring with the value of input array.
        '''
        
        return 0.4 * np.exp(-0.5 * x ** 2)

    @staticmethod
    def cdf(x: np.ndarray) -> np.ndarray:
        '''
        Estimate the normal cumulative density function.

        Keyword Arguments:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: An estimation of the probability of a normally distributed random variable occurring with a value less than the input array.
        '''

        return 0.5 * (1 + np.tanh(1.142 * x * (1 + 0.043595 * x ** 2)))

    def selectAction(self) -> int:
        '''
        Randomly select action with probabilities calculated.

        Returns:
            int: Index of action selected.
        '''

        stds = np.sqrt((self.sumsSquared - self.sums ** 2 / self.wSum) / self.wSum ** 2)

        normXValues = (self.xValues[:, None] - self.sums / self.wSum) / stds

        pdfs = self.pdf(normXValues) / stds
        cdfs = self.cdf(normXValues)

        cdfProd = np.prod(cdfs, axis = 1, keepdims = True) / (cdfs + 1e-10)

        ps = np.trapz(y = pdfs * cdfProd, x = self.xValues, axis = 0)
        ps = np.maximum(ps, 1e-10)
        ps /= sum(ps)

        if np.any(np.isnan(ps)):
            return np.random.choice(self.numActions)

        return np.random.choice(self.numActions, p = ps)
    
    def update(self, reward: float, action: int) -> None:
        '''
        Update internal state using action selected and reward received.

        Keyword Arguments:
            reward (float): Reward received from selecting action.
            action (int): Index of action selected.
        '''

        self.sumsSquared[action] = self.discount * self.sumsSquared[action] + reward ** 2
        self.sums[action] = self.discount * self.sums[action] + reward
        self.wSum[action] = self.discount * self.wSum[action] + 1

        self.rewards.append(reward)
        self.actions.append(action)