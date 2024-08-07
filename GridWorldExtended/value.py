from GridWorld import GridWorld
from tqdm import tqdm
import numpy as np

ALPHA = 0.01
GAMMA = 0.9

env = GridWorld(7, 7, defReward = 0, fallPenalty = 0)

initState = (3, 3)

env.addTerminalState(6, 0)
env.addTerminalState(0, 6)

env[(5, 0)] = {a: {((6, 0), -1): 1} if a == 1 else env[(5, 0)].map[a] for a in range(4)}
env[(6, 1)] = {a: {((6, 0), -1): 1} if a == 2 else env[(6, 1)].map[a] for a in range(4)}

env[(0, 5)] = {a: {((0, 6), 1): 1} if a == 3 else env[(0, 5)].map[a] for a in range(4)}
env[(1, 6)] = {a: {((0, 6), 1): 1} if a == 0 else env[(1, 6)].map[a] for a in range(4)}

rewards = np.array([-1, 0, 1]).reshape(1, 1, 1, 1, 1, -1)
probabilities = np.zeros((7, 7, 7, 7, 4, 3))
policy = np.full((7, 7, 1, 1, 4, 1), 0.25)

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

def features(state: tuple) -> np.ndarray:
    '''
    Converts state from tuple form to np.ndarray form.

    Keyword Arguments:
        state (tuple): Input state in tuple form.

    Returns:
        np.ndarray: Input state but in np.ndarray form.
    '''
    
    features = np.zeros((env.height, env.width))
    features[state] = 1

    return features

def gradMonteCarlo(env: GridWorld, iterations: int = 10_000) -> np.ndarray:
    '''
    Implement Gradient Monte Carlo algorithm for estimating value function.

    Keyword Arguments:
        env (GridWorld): The GridWorld environment.
        iterations (int): Number of episodes to run.

    Returns:
        np.ndarray: Estimated value function.
    '''

    w = np.zeros((env.height, env.width))
    for _ in tqdm(range(iterations)):
        state = initState
        episode = []
        
        while not env.isTerminal(state):
            action = np.random.randint(4)
            nextState, reward = env[state][action]
            episode.append((state, reward))
            state = nextState
        
        G = 0
        for (state, reward) in reversed(episode):
            G = GAMMA * G + reward
            
            x = features(state)
            w += ALPHA * (G - np.sum(w * x)) * x
        
    return w

def semiGradTDZero(env: GridWorld, iterations: int = 10_000) -> np.ndarray:
    '''
    Implement Semi-Gradient TD(0) algorithm for estimating value function.

    Keyword Arguments:
        env (GridWorld): The GridWorld environment.
        iterations (int): Number of episodes to run.

    Returns:
        np.ndarray: Estimated value function.
    '''
    
    w = np.zeros((env.height, env.width))
    for _ in tqdm(range(iterations)):
        state = initState
        
        while not env.isTerminal(state):
            action = np.random.randint(4)
            nextState, reward = env[state][action]

            x = features(state)
            xNext = features(nextState)
            
            w += ALPHA * (reward + GAMMA * np.sum(w * xNext) - np.sum(w * x)) * x
            
            state = nextState

    return w

if __name__ == '__main__':
    print('>>> Explicit solution:')
    values = explicitSolution(policy, probabilities, rewards).round(1)
    print(values)
    env.plotStateValueFunction(values)

    print(f'>>> Gradient Monte Carlo:')
    values = gradMonteCarlo(env).round(1)
    print(values)
    env.plotStateValueFunction(values)

    print(f'>>> Semi Gradient TD(0):')
    values = semiGradTDZero(env).round(1)
    print(values)
    env.plotStateValueFunction(values)