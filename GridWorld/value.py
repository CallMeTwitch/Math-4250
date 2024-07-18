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
env[0, 4] = {a: {((3, 2), 2.5): 0.5, ((4, 4), 2.5): 0.5} for a in range(4)}

rewardsList = rewards[0, 0, 0, 0, 0].tolist()
for y, x, block in env:
    for action, map in block:
        for ((yPrime, xPrime), reward), probability in map.items():
            probabilities[y, x, yPrime, xPrime, action, rewardsList.index(reward)] = probability

def explicitSolution(policy: np.ndarray, probabilities: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    '''
    Calculate explicit solution to system of Bellman Equations for the value function.

    Keyword Arguments:
        policy (np.ndarray): Maps states to probabilities of next states.
        probabilities (np.ndarray): Probability of achieving reward and going to next state given action and current state.
        rewards (np.ndarray): Possible rewards.

    Returns:
        (np.ndarray): Value function.
    '''

    height, width = probabilities.shape[:2]
    P = probabilities.reshape(width * height, width * height, 4, -1)
    R = rewards.reshape(1, 1, 1, -1)
    policy = policy.copy().reshape(width * height, 1, 4, 1)

    P, R = np.sum(P * policy, axis = (2, 3)), np.sum(P * R * policy, axis = (1, 2, 3))

    I = np.eye(width * height)
    V = np.linalg.solve(I - GAMMA * P, R)

    return V.reshape(height, width)

def iterativePolicyEval(policy: np.ndarray, probabilities: np.ndarray, rewards: np.ndarray, iterations: int = 1_000, threshold: float = 1e-6) -> np.ndarray:
    '''
    Derive value function using iterative policy evaluation.

    Keyword Arguments:
        policy (np.ndarray): Maps states to probabilities of next states.
        probabilities (np.ndarray): Probability of achieving reward and going to next state given action and current state.
        rewards (np.ndarray): Possible rewards.
        iterations (int): Maximum number of iterations to perform.
        threshold (float): Threshold to initiate early stopping (convergence threshold.)

    Returns:
        (np.ndarray): Value function.
    '''

    height, width = probabilities.shape[:2]
    values = np.zeros((height, width, 1, 1, 1, 1))
    for iteration in range(iterations):
        newValues = (policy * probabilities * (rewards + GAMMA * values.transpose(2, 3, 0, 1, 4, 5))).sum(axis = (2, 3, 4, 5), keepdims = True)

        if np.max(np.abs(newValues - values)) < threshold:
            print(f'Converged after {iteration + 1} iterations.')
            break

        values = newValues

    else:
        print('Reached maximum iterations without convergence.')

    return values.reshape(height, width)

def valueIteration(probabilities: np.ndarray, rewards: np.ndarray, iterations: int = 1_000, threshold: float = 1e-6) -> np.ndarray:
    '''
    Derive optimal value function using value iteration.

    Keyword Arguments:
        probabilities (np.ndarray): Probability of achieving reward and going to next state given action and current state.
        rewards (np.ndarray): Possible rewards.
        iterations (int): Maximum number of iterations to perform.
        threshold (float): Threshold to initiate early stopping (convergence threshold.)

    Returns:
        (np.ndarray): Optimal value function.
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

    return values.reshape(height, width)

if __name__ == '__main__':
    print('>>> Explicit solution:')
    values = explicitSolution(policy, probabilities, rewards).round(1)
    print(values)
    env.plotStateValueFunction(values)

    print('\n>>> Iterative Policy Evaluation:')
    values = iterativePolicyEval(policy, probabilities, rewards).round(1)
    print(values)
    env.plotStateValueFunction(values)

    print('\n>>> Value iteration:')
    values = valueIteration(probabilities, rewards).round(1)
    print(values)
    env.plotStateValueFunction(values)
