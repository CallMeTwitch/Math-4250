import numpy as np

class Stationary:
    '''
    Simulates the bandits in the k-armed bandit environment with stationary normally distributed rewards.
    '''

    __slots__ = 'means',
    def __init__(self, numBandits: int = 10) -> None:
        '''
        Initialize stationary bandit.

        Keyword Arguments:
            numBandits (int): Number of bandits to simulate.
        '''
        
        self.means = np.random.normal(0, 1, numBandits)

    def __call__(self, action: int) -> float:
        '''
        Get reward.

        Keyword Arguments:
            action (int): Index of the action selected.

        Returns:
            float: A random normally distributed reward with mean from the 'action'-th bandit.
        '''
        
        return np.random.normal(self.means[action], 1)
    
class Drift:
    '''
    Simulates the bandits in the k-armed bandit environment with drifting normally distributed rewards.
    '''
    
    __slots__ = 'epsilon', 'means'
    def __init__(self, numBandits: int = 10, epsilon: float = 0.001) -> None:
        '''
       Initialize drifting bandit.

        Keyword Arguments:
            epsilon (float): Standard deviation of the drifting variable.
            numBandits (int): Number of bandits to simulate.
        '''

        self.means = np.random.normal(0, 1, numBandits)
        self.epsilon = epsilon

    def __call__(self, action: int) -> float:
        '''
        Update internal state and get reward.

        Keyword Arguments:
            action (int): Index of the action selected.

        Returns:
            float: A random normally distributed reward with mean from the 'action'-th bandit.
        '''
        
        self.means += np.random.normal(0, self.epsilon)

        return np.random.normal(self.means[action], 1)
    
class MeanReverting:
    '''
    Simulates the bandits in the k-armed bandit environment with mean-reverting normally distributed rewards.
    '''

    __slots__ = 'epsilon', 'means'
    def __init__(self, numBandits: int = 10, epsilon: float = 0.01) -> None:
        '''
       Initialize mean-reverting bandit.

        Keyword Arguments:
            epsilon (float): Standard deviation of the random variable.
            numBandits (int): Number of bandits to simulate.
        '''

        self.means = np.random.normal(0, 1, numBandits)
        self.epsilon = epsilon

    def __call__(self, action: int) -> float:
        '''
        Update internal state and get reward.

        Keyword Arguments:
            action (int): Index of the action selected.

        Returns:
            float: A random normally distributed reward with mean = self.mean.
        '''
        
        self.means = 0.5 * self.means + np.random.normal(0, self.epsilon)

        return np.random.normal(self.means[action], 1)
    
class Abrupt:
    '''
    Simulates the bandits in the k-armed bandit environment with abruptly changing normally distributed rewards.
    '''

    __slots__ = 'epsilon', 'means'
    def __init__(self, numBandits: int = 10, epsilon: float = 0.005) -> None:
        '''
       Initialize abruptly changing bandit.

        Keyword Arguments:
            epsilon (float): Decimal robability of permuting mean rewards.
            numBandits (int): Number of bandits to simulate.
        '''

        self.means = np.random.normal(0, 1, numBandits)
        self.epsilon = epsilon

    def __call__(self, action: int) -> float:
        '''
        Update internal state and get reward.

        Keyword Arguments:
            action (int): Index of the action selected.

        Returns:
            float: A random normally distributed reward with mean = self.mean.
        '''
        
        if np.random.rand() < self.epsilon:
            self.means = np.random.permutation(self.means).tolist()

        return np.random.normal(self.means[action], 1)