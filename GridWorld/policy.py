from scipy.optimize import minimize
from GridWorld import GridWorld
import numpy as np

# Dimensions: (state column, state row, next state column, next state row, action, reward)
# Dimensions: (5 x 5 x 5 x 5 x 4 x 4)

GAMMA = 0.95

rewards = np.array([-0.5, 0, 2.5, 5]).reshape(1, 1, 1, 1, 1, -1)
probabilities = np.zeros((5, 5, 5, 5, 4, 4))
policy = np.full((5, 5, 1, 1, 4, 1), 0.25)

env = GridWorld(width = 5, height = 5, fallPenalty = -0.5)

env[0, 1] = {a: {((3, 2), 5): 1} for a in range(4)}
env[0, 4] = {a: {((2, 3), 2.5): 0.5, ((4, 4), 2.5): 0.5} for a in range(4)}

rewardsList = rewards[0, 0, 0, 0, 0].tolist()   
for y, x, block in env:
    for action, map in block:
        for ((yPrime, xPrime), reward), probability in map.items():
            probabilities[y, x, yPrime, xPrime, action, rewardsList.index(reward)] = probability

def explicitSolution(probabilities: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    '''
    Calculate explicit solution to Bellman Optimality Equation for the optimal policy function.

    Keyword Arguments:
        probabilities (np.ndarray): Probability of achieving reward and going to next state given action and current state.
        rewards (np.ndarray): Possible rewards.

    Returns:
        (np.ndarray): Optimal policy.
    '''
    
    height, width = probabilities.shape[:2]
    def loss(values: np.ndarray) -> float:
        '''
        Calculates MSE for value function.

        Keyword Arguments:
            values (np.ndarray): Estimate of value function.

        Returns:
            (np.ndarray): Approximation of value function
        '''

        values = values.reshape((height, width, 1, 1, 1, 1))
        newValues = (probabilities * (rewards + GAMMA * values.transpose(2, 3, 0, 1, 4, 5))).sum(axis = (2, 3, 5), keepdims = True).max(axis = 4, keepdims = True)
        return np.mean(np.square(values - newValues))
    
    values = minimize(loss, np.random.rand(height * width), method = 'L-BFGS-B').x.reshape(height, width, 1, 1, 1, 1)
    Q = (probabilities * (rewards + GAMMA * values.transpose(2, 3, 0, 1, 4, 5))).sum(axis = (2, 3, 5), keepdims = False)
    policy = Q.argmax(axis = -1, keepdims = False)
    return Q, policy

def policyIteration(policy: np.ndarray, probabilities: np.ndarray, rewards: np.ndarray, iterations: int = 1_000, threshold: float = 1e-6) -> np.ndarray:
    '''
    Derive optimal policy function using policy iteration.

    Keyword Arguments:
        policy (np.ndarray): Maps states to probabilities of next states.
        probabilities (np.ndarray): Probability of achieving reward and going to next state given action and current state.
        rewards (np.ndarray): Possible rewards.
        iterations (int): Maximum number of iterations to perform.
        threshold (float): Threshold to initiate early stopping (convergence threshold.)

    Returns:
        (np.ndarray): Optimal policy.
    '''

    height, width = probabilities.shape[:2]
    values = np.zeros((height, width, 1, 1, 1, 1))
    policy = policy.copy()

    for iteration in range(iterations):
        for _ in range(iterations):
            newValues = (policy * probabilities * (rewards + GAMMA * values.transpose(2, 3, 0, 1, 4, 5))).sum(axis = (2, 3, 4, 5), keepdims = True)

            if np.max(np.abs(newValues - values)) < threshold:
                break

            values = newValues

        Q = (probabilities * (rewards + GAMMA * values.transpose(2, 3, 0, 1, 4, 5))).sum(axis = (2, 3, 5), keepdims = True)
        newPolicy = np.eye(Q.shape[4])[np.argmax(Q, axis = 4)].reshape(list(Q.shape))

        if np.max(np.abs(newPolicy - policy)) < threshold:
            print(f'Converged after {iteration + 1} iterations.')
            break

        policy = newPolicy

    else:
        print('Reached maximum iterations without convergence.')
    
    return Q.reshape(height, width, 4), Q.argmax(axis = 4).reshape(height, width)

def valueIteration(probabilities: np.ndarray, rewards: np.ndarray, iterations: int = 1_000, threshold: float = 1e-6) -> np.ndarray:
    '''
    Derive optimal policy function using value iteration.

    Keyword Arguments:
        probabilities (np.ndarray): Probability of achieving reward and going to next state given action and current state.
        rewards (np.ndarray): Possible rewards.
        iterations (int): Maximum number of iterations to perform.
        threshold (float): Threshold to initiate early stopping (convergence threshold.)

    Returns:
        (np.ndarray): Optimal policy.
    '''

    height, width = probabilities.shape[:2]
    values = np.zeros((height, width, 1, 1, 1, 1))
    for iteration in range(iterations):
        newValues = (probabilities * (rewards + GAMMA * values.transpose(2, 3, 0, 1, 4, 5))).sum(axis = (2, 3, 5), keepdims = True).max(axis = 4, keepdims = True)

        if np.max(np.abs(newValues - values)) < threshold:
            print(f'Converged after {iteration + 1} iterations.')
            break
        
        values = newValues

    else:
        print('Reached maximum iterations without convergence.')

    Q = (probabilities * (rewards + GAMMA * values.transpose(2, 3, 0, 1, 4, 5))).sum(axis = (2, 3, 5), keepdims = False)
    policy = Q.argmax(axis = -1, keepdims = False)

    return Q, policy

arrows = np.array(['↑', '↓', '←', '→'])
if __name__ == '__main__':
    print('>>> Explicit solution:')
    Q, newPolicy = explicitSolution(probabilities, rewards)
    print(arrows[newPolicy])
    env.plotActionValueFunction(Q)

    print('\n>>> Policy iteration:')
    Q, newPolicy = policyIteration(policy, probabilities, rewards)
    print(arrows[newPolicy])
    env.plotActionValueFunction(Q)

    print('\n>>> Value iteration:')
    Q, newPolicy = valueIteration(probabilities, rewards)
    print(arrows[newPolicy])
    env.plotActionValueFunction(Q)
