from collections import defaultdict
from GridWorld import GridWorld
from tqdm import tqdm
import numpy as np

# Dimensions: (state column, state row, next state column, next state row, action, reward)
# Dimensions: (5 x 5 x 5 x 5 x 4 x 1)

GAMMA = 0.95

policy = np.full((5, 5, 1, 1, 4, 1), 0.25)

env = GridWorld(width = 5, height = 5, defReward = -0.2, fallPenalty = -0.5)

env[0, 1] = {a: {((4, 2), 5): 1} for a in range(4)}
env[0, 4] = {a: {((4, 2), 2.5): 0.5, ((4, 4), 2.5): 0.5} for a in range(4)}

env.addTerminalState(4, 0)
env.addTerminalState(2, 4)

def monteCarloExploringStarts(env: GridWorld, episodes: int = 10_000) -> np.ndarray:
    '''
    Derive optimal policy function using monte carlo with exploring starts.

    Keyword Arguments:
        env (GridWorld): Grid world environment.
        episodes (int): number of episodes.
    
    Returns:
        (np.ndarray): Optimal policy.
    '''

    height, width = env.height, env.width
    Q = np.zeros((height, width, 4), dtype = np.float32)
    R = defaultdict(list)

    for _ in tqdm(range(episodes)):
        state = (np.random.randint(low = 0, high = height - 1), np.random.randint(low = 0, high = width - 1))

        episode = []
        while True:
            action = np.random.randint(4)
            nextState, reward = env[state][action]

            episode.append((state, action, reward))
            if env.isTerminal(state):
                break

            state = nextState

        G = 0
        for t, (state, action, reward) in list(enumerate(episode))[::-1]:
            G = GAMMA * G + reward

            if state not in list(zip(*episode))[0][:t]:
                R[(state, action)].append(G)
                Q[state[0], state[1], action] = np.mean(R[(state, action)])

    return Q, Q.argmax(axis = 2)

def monteCarloEpsilonSoft(env: GridWorld, episodes: int = 10_000, epsilon: float = 0.5) -> np.ndarray:
    '''
    Derive optimal policy function using monte carlo with epsilon soft policy.

    Keyword Arguments:
        env (GridWorld): Grid world environment.
        episodes (int): number of episodes.
        epsilon (float): Decimal percent frequency of random policy actions.
    
    Returns:
        (np.ndarray): Optimal policy.
    '''

    height, width = env.height, env.width
    Q = np.zeros((height, width, 4), dtype = np.float32)
    R = defaultdict(list)

    initState = (np.random.randint(low = 0, high = height - 1), np.random.randint(low = 0, high = width - 1))
    while env.isTerminal(tuple(initState)):
        initState = (np.random.randint(low = 0, high = height - 1), np.random.randint(low = 0, high = width - 1))

    for _ in tqdm(range(episodes)):
        state = initState

        episode = []
        while True:
            action = np.random.randint(4) if np.random.random() < epsilon else np.argmax(Q[state])
            nextState, reward = env[state][action]

            episode.append((state, action, reward))
            if env.isTerminal(state):
                break

            state = nextState

        G = 0
        for t, (state, action, reward) in list(enumerate(episode))[::-1]:
            G = GAMMA * G + reward

            if state not in list(zip(*episode))[0][:t]:
                R[(state, action)].append(G)
                Q[state[0], state[1], action] = np.mean(R[(state, action)])

    return Q, Q.argmax(axis = 2)

def monteCarloOffPolicy(behaviourPolicy: np.ndarray, env: GridWorld, episodes: int = 10_000) -> np.ndarray:
    '''
    Derive optimal policy function using monte carlo off policy (using behaviour policy).

    Keyword Arguments:
        behaviourPolicy (np.ndarray): Behaviour policy.
        env (GridWorld): Grid world environment.
        episodes (int): number of episodes.
    
    Returns:
        (np.ndarray): Optimal policy.
    '''

    height, width = behaviourPolicy.shape[:2]
    Q = np.zeros((height, width, 4), dtype = np.float32)
    C = np.zeros((height, width, 4), dtype = np.float32)
    for _ in tqdm(range(episodes)):
        state = (np.random.randint(low = 0, high = height), np.random.randint(low = 0, high = width))

        episode = []
        while True:
            action = np.random.choice(4, p = behaviourPolicy[state].reshape(4))
            nextState, reward = env[state][action]

            episode.append((state, action, reward))

            if env.isTerminal(state):
                break

            state = nextState

        G, W = 0, 1
        for state, action, reward in episode[::-1]:
            G = GAMMA * G + reward

            C[state[0], state[1], action] += W
            Q[state[0], state[1], action] += (W / C[state[0], state[1], action]) * (G - Q[state[0], state[1], action])

            if action != Q[state].argmax():
                W *= behaviourPolicy[state].reshape(4)[action]
            else:
                continue

    return Q, Q.argmax(axis = 2)

arrows = np.array(['↑', '↓', '←', '→'])
if __name__ == '__main__':
    print('>>> Monte Carlo with Exploring Starts:')
    Q, newPolicy = monteCarloExploringStarts(env)
    env.plotActionValueFunction(Q)
    print(arrows[newPolicy])

    print('\n>>> Monte Carlo with Epsilon Soft Policy:')
    Q, newPolicy = monteCarloEpsilonSoft(env)
    env.plotActionValueFunction(Q)
    print(arrows[newPolicy])

    print('\n>>> Off-Policy Monte Carlo:')
    Q, newPolicy = monteCarloOffPolicy(policy, env)
    env.plotActionValueFunction(Q)
    print(arrows[newPolicy])
