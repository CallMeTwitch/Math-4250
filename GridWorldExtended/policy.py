from matplotlib import pyplot as plt
from GridWorld import GridWorld
from typing import Tuple
from tqdm import tqdm
import numpy as np

EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
EPSILON = 1.0
ALPHA = 0.1
GAMMA = 0.9

env = GridWorld(5, 5, defReward = -1, fallPenalty = -1)

initState = (4, 0)

env[(2, 0)] = {a: {(initState, -20): 1} for a in range(4)}
env[(2, 1)] = {a: {(initState, -20): 1} for a in range(4)}
env[(2, 3)] = {a: {(initState, -20): 1} for a in range(4)}
env[(2, 4)] = {a: {(initState, -20): 1} for a in range(4)}

env.addTerminalState(0, 0)
env.addTerminalState(0, 4)

rewards = np.array([-20, -1, 0]).reshape(1, 1, 1, 1, 1, -1)
probabilities = np.zeros((5, 5, 5, 5, 4, 3))

rewardsList = rewards[0, 0, 0, 0, 0].tolist()   
for y, x, block in env:
    for action, map in block:
        for ((yPrime, xPrime), reward), probability in map.items():
            probabilities[y, x, yPrime, xPrime, action, rewardsList.index(reward)] = probability

def valueIteration(probabilities: np.ndarray, rewards: np.ndarray, iterations: int = 1_000, threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Derive optimal policy function using value iteration.

    Keyword Arguments:
        probabilities (np.ndarray): Probability of achieving reward and going to next state given action and current state.
        rewards (np.ndarray): Possible rewards.
        iterations (int): Maximum number of iterations to perform.
        threshold (float): Threshold to initiate early stopping (convergence threshold.)

    Returns:
        (np.ndarray): Optimal action-value function.
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

def sarsa(env: GridWorld, iterations: int = 5_000) -> Tuple[np.ndarray, list]:
    '''
    Derive optimal action-value function using SARSA.

    Keyword Arguments:
        env (GridWorld): The GridWorld environment in which to perform SARSA.
        iterations (int): Number of iterations to perform.

    Returns:
        (np.ndarray): Optimal action-value function.
        (list): List of rewards per episode.
    '''
    
    Q = np.zeros((env.height, env.width, 4))
    rewards = []

    epsilon = EPSILON
    for _ in tqdm(range(iterations)):
        state = initState

        totalReward = 0
        action = np.random.randint(4) if np.random.random() < epsilon else np.argmax(Q[state])
        while not env.isTerminal(state):
            nextState, reward = env[state][action]
            nextAction = np.random.randint(4) if np.random.random() < epsilon else np.argmax(Q[nextState])

            Q[state][action] += ALPHA * (reward + GAMMA * Q[nextState][nextAction] - Q[state][action])

            state, action = nextState, nextAction
            totalReward += reward

        rewards.append(totalReward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return Q, rewards

def qLearning(env: GridWorld, iterations: int = 5_000) -> Tuple[np.ndarray, list]:
    '''
    Derive optimal action-value function using Q-Learning.

    Keyword Arguments:
        env (GridWorld): The GridWorld environment in which to perform Q-Learning.
        iterations (int): Number of iterations to perform.

    Returns:
        (np.ndarray): Optimal action-value function.
        (list): List of rewards per episode.
    '''
        
    Q = np.zeros((env.height, env.width, 4))
    rewards = []

    epsilon = EPSILON
    for _ in tqdm(range(iterations)):
        state = initState

        totalReward = 0
        while not env.isTerminal(state):
            action = np.random.randint(4) if np.random.random() < epsilon else np.argmax(Q[state])
            nextState, reward = env[state][action]

            Q[state][action] += ALPHA * (reward + GAMMA * np.max(Q[nextState]) - Q[state][action])

            state = nextState
            totalReward += reward

        rewards.append(totalReward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return Q, rewards

def plotTrajectory(env: GridWorld, Q: np.ndarray) -> None:
    '''
    Plots a sample trajectory in the given environment using the action-value function.

    Keyword Arguments:
        env (GridWorld): Environment to sample trajectory in.
        Q (np.ndarray): Action-value function to use for trajectory.
    '''
    
    trajectory = [initState]

    while not env.isTerminal(state := trajectory[-1]):
        action = np.argmax(Q[state])
        nextState, _ = env[state][action]
        trajectory.append(nextState)

    y, x = zip(*trajectory)

    plt.figure(figsize = (8, 8))
    plt.plot(x, y, 'bo-')
    plt.xlim(-0.5, env.width - 0.5)
    plt.ylim(env.height - 0.5, -0.5)
    plt.grid(True)
    plt.title('Agent Trajectory')
    plt.show()

def plotRewards(sarsaRewards: list, qRewards: list) -> None:
    '''
    Plot rewards per episode for both SARSA and Q-Learning algorithms.

    Keyword Arguments;
        sarsaRewards (list): List of rewards per epsiode gathered from the SARSA algorithm.
        qRewards (list): List of rewards per epsiode gathered from the Q-Learning algorithm.
    '''
    
    plt.figure(figsize = (10, 6))
    plt.plot(sarsaRewards, label=  'Sarsa')
    plt.plot(qRewards, label = 'Q-learning')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Sum of Rewards per Episode')
    plt.legend()
    plt.show()

arrows = np.array(['↑', '↓', '←', '→'])
if __name__ == '__main__':
    print('>>> Value Iteration:')
    Q, policy = valueIteration(probabilities, rewards)
    print(arrows[policy])
    env.plotActionValueFunction(Q)

    print('>>> SARSA:')
    Q, sarsaRewards = sarsa(env)
    print(arrows[Q.argmax(axis = 2)])
    env.plotActionValueFunction(Q)
    plotTrajectory(env, Q)

    print('\n>>> Q-Learning:')
    Q, qRewards = qLearning(env)
    print(arrows[Q.argmax(axis = 2)])
    env.plotActionValueFunction(Q)
    plotTrajectory(env, Q)

    plotRewards(sarsaRewards, qRewards)