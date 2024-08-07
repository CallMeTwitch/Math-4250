from matplotlib import animation as anim
from matplotlib import pyplot as plt
from GridWorld import GridWorld
from typing import Tuple
from tqdm import tqdm
import numpy as np

EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
EPSILON = 1.0
GAMMA = 0.9
ALPHA = 0.1

env = GridWorld(5, 5, defReward = 0, fallPenalty = -1)

predInitState = (env.height - 1, 0)
preyInitState = (0, env.width - 1)

def sarsa(env: GridWorld, iterations: int = 5_000) -> Tuple[np.ndarray, list]:
    '''
    Derive optimal action-value function using SARSA.

    Keyword Arguments:
        env (GridWorld): The GridWorld environment in which to perform SARSA.
        iterations (int): Maximum number of iterations to perform.

    Returns:
        (np.ndarray): Optimal action-value function.
        (list): List of rewards per episode.
    '''
    
    predQ = np.zeros((env.height, env.width, env.height, env.width, 4))
    preyQ = np.zeros((env.height, env.width, env.height, env.width, 4))
    predRewards = []
    preyRewards = []

    epsilon = EPSILON
    for _ in tqdm(range(iterations)):
        predState = predInitState
        preyState = preyInitState

        totalPredReward = 0
        totalPreyReward = 0

        predAction = np.random.randint(4) if np.random.random() < epsilon else np.argmax(predQ[predState][preyState])
        preyAction = np.random.randint(4) if np.random.random() < epsilon else np.argmax(preyQ[predState][preyState])
        for _ in range(iterations):
            predNextState, predReward = env[predState][predAction]
            preyNextState, preyReward = env[preyState][preyAction]

            if predState == preyState:
                predReward += 1
                preyReward -= 1

            predNextAction = np.random.randint(4) if np.random.random() < epsilon else np.argmax(predQ[predNextState][preyNextState])
            preyNextAction = np.random.randint(4) if np.random.random() < epsilon else np.argmax(preyQ[predNextState][preyNextState])

            predQ[predState][preyState][predAction] += ALPHA * (predReward + GAMMA * predQ[predNextState][preyNextState][predNextAction] - predQ[predState][preyState][predAction])
            preyQ[predState][preyState][preyAction] += ALPHA * (preyReward + GAMMA * preyQ[predNextState][preyNextState][preyNextAction] - preyQ[predState][preyState][preyAction])

            predState, predAction = predNextState, predNextAction
            preyState, preyAction = preyNextState, preyNextAction

            totalPredReward += predReward
            totalPreyReward += preyReward

        predRewards.append(totalPredReward)
        preyRewards.append(totalPreyReward)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return predQ, preyQ, predRewards, preyRewards

def qLearning(env: GridWorld, iterations: int = 5_000) -> Tuple[np.ndarray, list]:
    '''
    Derive optimal action-value function using Q-Learning.

    Keyword Arguments:
        env (GridWorld): The GridWorld environment in which to perform SARSA.
        iterations (int): Maximum number of iterations to perform.

    Returns:
        (np.ndarray): Optimal action-value function.
        (list): List of rewards per episode.
    '''

    predQ = np.zeros((env.height, env.width, env.height, env.width, 4))
    preyQ = np.zeros((env.height, env.width, env.height, env.width, 4))
    predRewards = []
    preyRewards = [] 

    epsilon = EPSILON
    for _ in tqdm(range(iterations)):
        predState = predInitState
        preyState = preyInitState

        totalPredReward = 0
        totalPreyReward = 0
        for _ in range(iterations):
            predAction = np.random.randint(4) if np.random.random() < epsilon else np.argmax(predQ[predState][preyState])
            preyAction = np.random.randint(4) if np.random.random() < epsilon else np.argmax(preyQ[predState][preyState])

            nextPredState, predReward = env[predState][predAction]
            nextPreyState, preyReward = env[preyState][preyAction]

            if predState == preyState:
                predReward += 1
                preyReward -= 1

            predQ[predState][preyState][predAction] += ALPHA * (predReward + GAMMA * np.max(predQ[nextPredState][nextPreyState]) - predQ[predState][preyState][predAction])
            preyQ[predState][preyState][preyAction] += ALPHA * (preyReward + GAMMA * np.max(preyQ[nextPredState][nextPreyState]) - preyQ[predState][preyState][preyAction])

            predState = nextPredState
            preyState = nextPreyState

            totalPredReward += predReward
            totalPreyReward += preyReward

        predRewards.append(totalPredReward)
        preyRewards.append(totalPreyReward)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return predQ, preyQ, predRewards, preyRewards

def plotRewards(predatorRewards: list, preyRewards: list) -> None:
    '''
    Plot rewards for predator vs prey.

    Keyword Arguments:
        predatorRewards (list): List of rewards per episode for predator.
        preyRewards (list): List of rewards per episode for prey.
    '''

    plt.figure(figsize = (12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(predatorRewards, c = 'k')
    plt.title('Predator Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(preyRewards, c = 'k')
    plt.title('Prey Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.tight_layout()
    plt.show()

def animateOptimalEpisode(env: GridWorld, predQ: np.ndarray, preyQ: np.ndarray, maxSteps: int = 50, title: str = '') -> None:
    '''
    Animate an optimal episode between predator and prey.

    Keyword Arguments:
        env (GridWorld): GridWorld environment to perform episode.
        predQ (np.ndarray): Action-value function for predator.
        preyQ (np.ndarray): Action-value function for prey.
        maxSteps (int): maximum number of steps to perform in episode.
        title (str): Name for saved video.
    '''

    predPath = [predInitState]
    preyPath = [preyInitState]
    for _ in range(maxSteps):
        predAction = np.argmax(predQ[predPath[-1]][preyPath[-1]])
        preyAction = np.argmax(preyQ[predPath[-1]][preyPath[-1]])

        predPath.append(env[predPath[-1]][predAction][0])
        preyPath.append(env[preyPath[-1]][preyAction][0])

    fig, ax = plt.subplots(figsize = (8, 8))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.grid(True)
    ax.set_title('Predator-Prey Optimal Episode')

    pred, = ax.plot([], [], 'ro', markersize = 10, label = 'Predator')
    prey, = ax.plot([], [], 'bo', markersize = 10, label = 'Prey')
    ax.legend()

    def init():
        pred.set_data([], [])
        prey.set_data([], [])
        return pred, prey

    def animate(i):
        pathIndex = i // 10
        stepFraction = (i % 10) / 10.0

        if pathIndex < len(predPath) - 1:
            predX = predPath[pathIndex][1] + stepFraction * (predPath[pathIndex + 1][1] - predPath[pathIndex][1])
            predY = predPath[pathIndex][0] + stepFraction * (predPath[pathIndex + 1][0] - predPath[pathIndex][0])
            preyX = preyPath[pathIndex][1] + stepFraction * (preyPath[pathIndex + 1][1] - preyPath[pathIndex][1])
            preyY = preyPath[pathIndex][0] + stepFraction * (preyPath[pathIndex + 1][0] - preyPath[pathIndex][0])
            pred.set_data([predX], [predY])
            prey.set_data([preyX], [preyY])

        return pred, prey
    
    numFrames = (len(predPath) - 1) * 10
    a = anim.FuncAnimation(fig, animate, init_func = init, frames = numFrames, interval = 100, blit = True)
    a.save(f'./{title}.gif', writer = 'pillow', fps = 10)
    plt.show()

    plt.figure(figsize = (8, 8))
    plt.imshow(np.zeros((env.height, env.width)), cmap = 'binary')
    plt.grid(True, which = 'both', color = 'gray', linestyle = '-', linewidth = 0.5)
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    
    predY, predX = zip(*predPath)
    preyY, preyX = zip(*preyPath)
    
    plt.plot(predX, predY, 'ro-', label = 'Predator', markersize = 8)
    plt.plot(preyX, preyY, 'bo-', label = 'Prey', markersize = 8)
    
    plt.title('Optimal Episode')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    print('>>> SARSA:')
    predQ, preyQ, predRewards, preyRewards = sarsa(env)
    plotRewards(predRewards, preyRewards)
    animateOptimalEpisode(env, predQ, preyQ, title = 'SARSA')

    print('>>> Q-Learning:')
    predQ, preyQ, predRewards, preyRewards = qLearning(env)
    plotRewards(predRewards, preyRewards)
    animateOptimalEpisode(env, predQ, preyQ, title = 'Q-Learning')