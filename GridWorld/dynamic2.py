from GridWorld import GridWorld
import numpy as np

# Dimensions: (time step, state column, state row, next state column, next state row, action, reward)
# Dimensions: (MAX_EPISODE_LENGTH x 5 x 5 x 5 x 5 x 4 x 5)

MAX_EPISODE_LENGTH = 1_000
GAMMA = 0.95

rewards = np.array([-0.5, -0.2, 0, 2.5, 5]).reshape(1, 1, 1, 1, 1, 1, -1)
probabilities = np.zeros((MAX_EPISODE_LENGTH, 5, 5, 5, 5, 4, 5))
policy = np.full((MAX_EPISODE_LENGTH, 5, 5, 1, 1, 4, 1), 0.25)

env = GridWorld(width = 5, height = 5, defReward = -0.2, fallPenalty = -0.5)

env[0, 1] = {a: {((4, 2), 5): 1} for a in range(4)}
env[0, 4] = {a: {((4, 2), 2.5): 0.5, ((4, 4), 2.5): 0.5} for a in range(4)}

env.addTerminalState(4, 0)
env.addTerminalState(2, 4)

rewardsList = rewards[0, 0, 0, 0, 0, 0].tolist()
for t in range(MAX_EPISODE_LENGTH):
    for y, x, block in env:
        for action, map in block:
            for ((yPrime, xPrime), reward), probability in map.items():
                if (y, x) == (0, 1):
                    if (yPrime, xPrime) == (4, 2) and reward == 5:
                        probabilities[t, y, x, yPrime, xPrime, action, rewardsList.index(reward)] = 1 * (1 + 0.8 ** t) / 2
                    elif (yPrime, xPrime) == (4, 2) and reward == 2.5:
                        probabilities[t, y, x, yPrime, xPrime, action, rewardsList.index(reward)] = 0.5 * (1 - 0.8 ** t) / 2
                    elif (yPrime, xPrime) == (4, 4) and reward == 2.5:
                        probabilities[t, y, x, yPrime, xPrime, action, rewardsList.index(reward)] = 0.5 * (1 - 0.8 ** t) / 2
                    else:
                        assert False
                elif (y, x) == (0, 4):
                    if (yPrime, xPrime) == (4, 2) and reward == 5:
                        probabilities[t, y, x, yPrime, xPrime, action, rewardsList.index(reward)] = 1 * (1 - 0.8 ** t) / 2
                    elif (yPrime, xPrime) == (4, 2) and reward == 2.5:
                        probabilities[t, y, x, yPrime, xPrime, action, rewardsList.index(reward)] = 0.5 * (1 + 0.8 ** t) / 2
                    elif (yPrime, xPrime) == (4, 4) and reward == 2.5:
                        probabilities[t, y, x, yPrime, xPrime, action, rewardsList.index(reward)] = 0.5 * (1 + 0.8 ** t) / 2
                    else:
                        assert False
                else:
                    probabilities[t, y, x, yPrime, xPrime, action, rewardsList.index(reward)] = probability

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

    height, width = probabilities.shape[1:3]
    values = np.zeros((MAX_EPISODE_LENGTH, height, width, 1, 1, 1, 1))
    policy = policy.copy()

    for iteration in range(iterations):
        for _ in range(iterations):
            newValues = (policy * probabilities * (rewards + GAMMA * values.transpose(0, 3, 4, 1, 2, 5, 6))).sum(axis = (3, 4, 5, 6), keepdims = True)

            if np.max(np.abs(newValues - values)) < threshold:
                break

            values = newValues

        Q = (probabilities * (rewards + GAMMA * values.transpose(0, 3, 4, 1, 2, 5, 6))).sum(axis = (3, 4, 6), keepdims = True)
        newPolicy = np.eye(Q.shape[5])[np.argmax(Q, axis = 5)].reshape(list(Q.shape))

        if np.max(np.abs(newPolicy - policy)) < threshold:
            print(f'Converged after {iteration + 1} iterations.')
            break

        policy = newPolicy

    else:
        print('Reached maximum iterations without convergence.')
    
    return Q.reshape(MAX_EPISODE_LENGTH, height, width, 4), Q.argmax(axis = 5).reshape(MAX_EPISODE_LENGTH, height, width)

arrows = np.array(['↑', '↓', '←', '→'])
if __name__ == '__main__':
    print('\n>>> Policy iteration:')
    Q, newPolicy = policyIteration(policy, probabilities, rewards)

    for t in range(0, MAX_EPISODE_LENGTH, 100):
        print(f'Time step: {t}')
        print(arrows[newPolicy[t]])
        env.plotActionValueFunction(Q[t])
