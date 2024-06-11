from typing import Callable, Union, Tuple, List, Dict
from matplotlib import pyplot as plt
from bandits import *
from agents import *
from tqdm import tqdm
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

def run(agentClass: Callable, banditClass: Callable, numSimulations: int, numTimesteps: int) -> Tuple:
    '''
    Run simulations with agent and bandit classes.

    Keyword Arguments:
        banditClass (Callable): Function returning new bandit instance.
        agentClass (Callable): Function returning new agent instance.
        numTimesteps (int): Number of timesteps per simulation.
        numSimulations (int): Number of simulations to run.

    Returns:
        np.ndarray[float]: Rewards per timestep per simulation.
        np.ndarray[int]: Actions per timestep per simulation.
        np.ndarray[int]: Optimal actions per timestep per simulation.
        np.ndarray[float]: Worst rewards per timestep per simulation.
        np.ndarray[float]: Best rewards per timestep per simulation.
    '''
    
    rewards = np.empty((numSimulations, numTimesteps))
    actions = np.empty((numSimulations, numTimesteps))

    bestActions = np.empty((numSimulations, numTimesteps))
    worstRewards = np.empty((numSimulations, numTimesteps))
    bestRewards = np.empty((numSimulations, numTimesteps))

    for sim in tqdm(range(numSimulations)):
        np.random.seed(sim)

        agent = agentClass()
        bandits = banditClass()

        for step in range(numTimesteps):
            action = agent.selectAction()
            reward = bandits(action)
            agent.update(reward, action)

            rewards[sim, step] = reward
            actions[sim, step] = action

            bestActions[sim, step] = np.argmax(bandits.means)
            worstRewards[sim, step] = min(np.min(bandits.means), rewards[sim, step])
            bestRewards[sim, step] = max(np.max(bandits.means), rewards[sim, step])

    return rewards, actions, bestActions, worstRewards, bestRewards

def calculateMetrics(rewards: np.ndarray, actions: np.ndarray, bestActions: np.ndarray, bestRewards: np.ndarray, worstRewards: np.ndarray) -> Tuple:
    '''
    Calculate metrics from simulation data.

    Keyword Arguments:
        rewards (np.ndarray[float]): Rewards per timestep per simulation.
        actions (np.ndarray[int]): Actions per timestep per simulation.
        bestActions (np.ndarray[int]): Optimal actions per timestep per simulation.
        worstRewards (np.ndarray[float]): Worst rewards per timestep per simulation.
        bestRewards (np.ndarray[float]): Best rewards per timestep per simulation.

    Returns:
        np.ndarray[float]: Average rewards across simulations (per timestep).
        float: Average total reward per simulation.
        np.ndarray[float]: Percent of actions selected which were optimal across simulations (per timestep).
        float: Percent of all actions selected which were optimal.
        float: Average normalized total reward per simulation.
        np.ndarray[float]: Average reward across timesteps (per simulation).
    '''
    
    averageRewardPerTimestep = np.array(rewards).mean(0)
    averageAccumulatedReward = np.array(rewards).sum(1).mean()
    pctOptimalActionsPerTimestep = (np.array(actions) == np.array(bestActions)).mean(0)
    pctOptimalActions = pctOptimalActionsPerTimestep.mean()
    averageNormalizedAccumulatedReward = ((np.array(rewards).sum(1) - np.array(worstRewards).sum(1)) / (np.array(bestRewards).sum(1) - np.array(worstRewards).sum(1))).mean()
    averageTerminalRewardDistribution = np.array(rewards).mean(1)

    return averageRewardPerTimestep, averageAccumulatedReward, pctOptimalActionsPerTimestep, pctOptimalActions, averageNormalizedAccumulatedReward, averageTerminalRewardDistribution

def savePlots(agents: List, data: Dict, bandits: Callable, folder: str) -> None:
    '''
    Generate and save plots documenting performance data.

    Keyword Arguments:
        agents (List[Callable]): List of agents to plot data from.
        data (Dict[str, List[np.ndarray[float] | float]]): Dictionary full of metric data from agents.
        bandits (Callable): Bandits used to obtain data.
        folder (str): Folder to save plots in.
    '''
    
    print()

    try:
        os.mkdir(f'./{folder}')
    except:
        pass

    titles = []
    bandit = bandits()
    for i, agent in enumerate(agents):
        agent = agent()

        if not hasattr(agent, '__slots__'):
            agent.__slots__ = tuple()
        title = f'{agent.__class__.__name__}({", ".join(f"{key} = {getattr(agent, key)}" for key in agent.__slots__)})'
        titles.append(title)

        print(f'{title} (AAR = {data["Average Accumulated Reward"][i]:.3f}, POA = {data["Percent Optimal Actions"][i]:.3%}, ANAR = {data["Average Normalized Accumulated Reward"][i]:.3f})')

        fig, axes = plt.subplots(1, 2, figsize = (12, 5))
        fig.suptitle(title, fontsize = 16)

        axes[0].set_title('Average Reward per Timestep')
        axes[0].plot(range(len(data['Average Reward per Timestep'][i])), data['Average Reward per Timestep'][i], color = 'k')
        axes[0].axhline(y = data['Average Terminal Reward Distribution'][i].mean(), color = 'r', linestyle = '--')
        axes[0].set_ylabel('Average Reward')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylim(bottom = -0.5, top = 2.0)

        axes[1].set_title('Percent Optimal Actions per Timestep')
        axes[1].plot(range(len(data['Percent Optimal Actions per Timestep'][i])), data['Percent Optimal Actions per Timestep'][i], color = 'k')
        axes[1].axhline(y = data['Percent Optimal Actions'][i], color = 'r', linestyle = '--')
        axes[1].set_ylabel('Percent Optimal Actions')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylim(bottom = 0, top = 1.0)

        plt.tight_layout()
        plt.savefig(f'./{folder}/{title.split(" (")[0]} Timesteps.jpg')

    plt.figure(figsize = (10, 10))
    plt.xticks(rotation = 15)
    plt.boxplot(data['Average Terminal Reward Distribution'], labels = [title.split(" (")[0] for title in titles])
    plt.ylabel('Average Reward at Terminal Timestep')
    plt.xlabel('Agent Type')
    plt.title('Distribution of Average Reward at Terminal Timestep per Agent Type in Non-Stationary Environments')
    
    plt.savefig(f'./{folder}/{bandit.__class__.__name__}.jpg')

if __name__ == '__main__':
    print(r'%%%%% Section 1 %%%%%')

    bandits = lambda: Stationary(10)

    agents = [
        lambda: GreedyAgent(10),
        lambda: EpsilonGreedyAgent(10, epsilon = 0.1),
        lambda: OptimisticAgent(10, initValue = 5.0),
        lambda: GradientAgent(10, alpha = 0.2),
        lambda: CustomAgent(10, discount = 1.0)
    ]

    data = {
        'Average Normalized Accumulated Reward': [],
        'Average Terminal Reward Distribution': [],
        'Percent Optimal Actions per Timestep': [],
        'Average Reward per Timestep': [],
        'Average Accumulated Reward': [],
        'Percent Optimal Actions': []
    }

    for agent in agents:
        averageRewardPerTimestep, averageAccumulatedReward, pctOptimalActionsPerTimestep, pctOptimalActions, averageNormalizedAccumulatedReward, averageTerminalRewardDistribution = calculateMetrics(*run(agent, bandits, 1_000, 1_000))
        data['Average Normalized Accumulated Reward'].append(averageNormalizedAccumulatedReward)
        data['Average Terminal Reward Distribution'].append(averageTerminalRewardDistribution)
        data['Percent Optimal Actions per Timestep'].append(pctOptimalActionsPerTimestep)
        data['Average Reward per Timestep'].append(averageRewardPerTimestep)
        data['Average Accumulated Reward'].append(averageAccumulatedReward)
        data['Percent Optimal Actions'].append(pctOptimalActions)

    savePlots(agents, data, bandits, 'Stationary')

    print('\n%%%%% Section 2.1.1 %%%%%')
    
    bandits = lambda: Drift(10, epsilon = 0.001)

    agents = [
        lambda: EpsilonGreedyAgent(10, epsilon = 0.01, stepSize = 0.2),
        lambda: EpsilonGreedyAgent(10, epsilon = 0.01),
        lambda: OptimisticAgent(10, initValue = 2.0),
        lambda: GradientAgent(10, alpha = 0.5),
        lambda: CustomAgent(10, discount = 0.9)
    ]

    data = {
        'Average Normalized Accumulated Reward': [],
        'Average Terminal Reward Distribution': [],
        'Percent Optimal Actions per Timestep': [],
        'Average Reward per Timestep': [],
        'Average Accumulated Reward': [],
        'Percent Optimal Actions': []
    }

    for agent in agents:
        averageRewardPerTimestep, averageAccumulatedReward, pctOptimalActionsPerTimestep, pctOptimalActions, averageNormalizedAccumulatedReward, averageTerminalRewardDistribution = calculateMetrics(*run(agent, bandits, 1_000, 10_000))
        data['Average Normalized Accumulated Reward'].append(averageNormalizedAccumulatedReward)
        data['Average Terminal Reward Distribution'].append(averageTerminalRewardDistribution)
        data['Percent Optimal Actions per Timestep'].append(pctOptimalActionsPerTimestep)
        data['Average Reward per Timestep'].append(averageRewardPerTimestep)
        data['Average Accumulated Reward'].append(averageAccumulatedReward)
        data['Percent Optimal Actions'].append(pctOptimalActions)

    savePlots(agents, data, bandits, 'Drifting')

    print('\n%%%%% Section 2.1.2 %%%%%')
    
    bandits = lambda: MeanReverting(10, 0.01)

    agents = [
        lambda: EpsilonGreedyAgent(10, epsilon = 0.1, stepSize = 0.1),
        lambda: EpsilonGreedyAgent(10, epsilon = 0.1),
        lambda: OptimisticAgent(10, initValue = 5.0),
        lambda: GradientAgent(10, alpha = 0.2),
        lambda: CustomAgent(10, discount = 0.8)
    ]

    data = {
        'Average Normalized Accumulated Reward': [],
        'Average Terminal Reward Distribution': [],
        'Percent Optimal Actions per Timestep': [],
        'Average Reward per Timestep': [],
        'Average Accumulated Reward': [],
        'Percent Optimal Actions': []
    }

    for agent in agents:
        averageRewardPerTimestep, averageAccumulatedReward, pctOptimalActionsPerTimestep, pctOptimalActions, averageNormalizedAccumulatedReward, averageTerminalRewardDistribution = calculateMetrics(*run(agent, bandits, 1_000, 10_000))
        data['Average Normalized Accumulated Reward'].append(averageNormalizedAccumulatedReward)
        data['Average Terminal Reward Distribution'].append(averageTerminalRewardDistribution)
        data['Percent Optimal Actions per Timestep'].append(pctOptimalActionsPerTimestep)
        data['Average Reward per Timestep'].append(averageRewardPerTimestep)
        data['Average Accumulated Reward'].append(averageAccumulatedReward)
        data['Percent Optimal Actions'].append(pctOptimalActions)

    savePlots(agents, data, bandits, 'MeanReverting')

    print('\n%%%%% Section 2.2 %%%%%')
    
    bandits = lambda: Abrupt(10, 0.005)

    agents = [
        lambda: EpsilonGreedyAgent(10, epsilon = 0.1, stepSize = 0.2),
        lambda: EpsilonGreedyAgent(10, epsilon = 0.1),
        lambda: OptimisticAgent(10, initValue = 5.0),
        lambda: GradientAgent(10, alpha = 0.2),
        lambda: CustomAgent(10, discount = 0.8)
    ]

    data = {
        'Average Normalized Accumulated Reward': [],
        'Average Terminal Reward Distribution': [],
        'Percent Optimal Actions per Timestep': [],
        'Average Reward per Timestep': [],
        'Average Accumulated Reward': [],
        'Percent Optimal Actions': []
    }

    for agent in agents:
        averageRewardPerTimestep, averageAccumulatedReward, pctOptimalActionsPerTimestep, pctOptimalActions, averageNormalizedAccumulatedReward, averageTerminalRewardDistribution = calculateMetrics(*run(agent, bandits, 1_000, 10_000))
        data['Average Normalized Accumulated Reward'].append(averageNormalizedAccumulatedReward)
        data['Average Terminal Reward Distribution'].append(averageTerminalRewardDistribution)
        data['Percent Optimal Actions per Timestep'].append(pctOptimalActionsPerTimestep)
        data['Average Reward per Timestep'].append(averageRewardPerTimestep)
        data['Average Accumulated Reward'].append(averageAccumulatedReward)
        data['Percent Optimal Actions'].append(pctOptimalActions)

    savePlots(agents, data, bandits, 'Abrupt')
