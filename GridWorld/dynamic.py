from collections import defaultdict
from GridWorld import GridWorld
from tqdm import tqdm
import numpy as np

# Dimensions: (state column, state row, next state column, next state row, action, reward)
# Dimensions: (5 x 5 x 5 x 5 x 4 x 5)

MAX_EPISODE_LENGTH = 10_000
GAMMA = 0.95

rewards = np.array([-0.5, -0.2, 0, 2.5, 5]).reshape(1, 1, 1, 1, 1, -1)
probabilities = np.zeros((5, 5, 5, 5, 4, 5))
policy = np.full((5, 5, 1, 1, 4, 1), 0.25)

env = GridWorld(width = 5, height = 5, defReward = -0.2, fallPenalty = -0.5)

env[0, 1] = {a: {((4, 2), 5): 1} for a in range(4)}
env[0, 4] = {a: {((4, 2), 2.5): 0.5, ((4, 4), 2.5): 0.5} for a in range(4)}

env.addTerminalState(4, 0)
env.addTerminalState(2, 4)

def monteCarloSampling(iterations: int = 10_000) -> dict:
    '''
    Performs random episodes of environment and returns data.

    Keyword Arguments:
        iterations (int): Number of episodes to perform.

    Returns:
        (dict): Dictionary containing state-action-nextState-reward frequencies.
    '''
    
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    height, width = policy.shape[:2]

    for _ in tqdm(range(iterations)):
        state = (np.random.randint(low = 0, high = height), np.random.randint(low = 0, high = width))

        for _ in range(MAX_EPISODE_LENGTH):
            action = np.random.choice(4)

            nextState, reward = env[state][action]
            data[state][action][(nextState, reward)] += 1
        
            if np.random.random() < 0.1:
                env[0, 1].map, env[0, 4].map = env[0, 4].map, env[0, 1].map
        
            if env.isTerminal(nextState):
                break

            state = nextState

    for state in data:
        for action in data[state]:
            total = sum(data[state][action].values())
            for key in data[state][action]:
                data[state][action][key] /= total

    return data

def getProbabilities(env: GridWorld, data: dict) -> np.ndarray:
    '''
    Gets transition matrix for data dictionary provided by monteCarloSampling function.

    Keyword Arguments:
        env (GridWorld): Environment.
        data (dict): Dictionary provided by MonteCarloSampling function.

    Returns:
        (np.ndarray): Transition matrix.
    '''
    
    rewardsList = rewards[0, 0, 0, 0, 0].tolist()
    probabilities = np.zeros((env.height, env.width, env.height, env.width, 4, len(rewardsList)))

    for y, x, block in env:
        for action, map in block:
            for ((yPrime, xPrime), reward), prob in map.items():
                probabilities[y, x, yPrime, xPrime, action, rewardsList.index(reward)] = data[(y, x)][action][((yPrime, xPrime), reward)]

    return probabilities

def policyIteration(policy: np.ndarray, rewards: np.ndarray, iterations: int = 1_000, threshold: float = 1e-6) -> np.ndarray:
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

    data = monteCarloSampling()
    probabilities = getProbabilities(env, data)

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

arrows = np.array(['↑', '↓', '←', '→'])
if __name__ == '__main__':
    print('>>> Policy Iteration')
    Q, newPolicy = policyIteration(policy, rewards)
    env.plotActionValueFunction(Q)
    print(arrows[newPolicy])
