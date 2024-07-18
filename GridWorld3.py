from typing import Generator, ItemsView, Tuple
from matplotlib import pyplot as plt
from matplotlib import patches
import seaborn as sns
import numpy as np

class GridWorld:
    '''
    Simulates a Grid World environment for reinforcement learning.
    '''

    def __init__(self, width: int, height: int, defReward: float = 0, fallPenalty: float = -1) -> None:
        '''
        Initialize the Grid World environment.

        Keyword Arguments:
            width (int): Width of the grid.
            height (int): Height of the grid.
            defReward (float): Default reward for any action.
            fallPenalty (float): Penalty for actions resulting in falling off the grid.
        '''

        class Block:
            '''
            Represents a single block in the Grid World environment.
            '''

            def __init__(self, y: int, x: int) -> None:
                '''
                Initialize a block in the Grid World.

                Keyword Arguments:
                    y (int): Y-coordinate of the block.
                    x (int): X-coordinate of the block.
                '''

                # Given current state, action (0 = up, 1 = down, 2 = left, 3 = right) maps to {(newState, reward): probability}
                self.map = {
                    0: {((y - 1, x), defReward): 1} if y else {((y, x), fallPenalty): 1},
                    1: {((y + 1, x), defReward): 1} if y < height - 1 else {((y, x), fallPenalty): 1},
                    2: {((y, x - 1), defReward): 1} if x else {((y, x), fallPenalty): 1},
                    3: {((y, x + 1), defReward): 1} if x < width - 1 else {((y, x), fallPenalty): 1}
                }

            def setMap(self, map: dict) -> None:
                '''
                Set special map for block.

                Keyword Arguments:
                    map (dict): Special map to set for block.
                '''
                
                self.map = map.copy()

            def __getitem__(self, key: int) -> Tuple[Tuple[int, int], float]:
                '''
                Get new state and reward given action in current state.

                Keyword Arguments:
                    key (int): Action (0 = up, 1 = down, 2 = left, 3 = right)

                Returns:
                    Tuple[Tuple[int, int], float]: (New state, reward) chosen randomly with given probabilities.;
                '''
                
                keys, values = list(self.map[key].keys()), list(self.map[key].values())
                return keys[np.random.choice(range(len(keys)), p = values)]
            
            def __iter__(self) -> Generator[ItemsView[int, dict], None, None]:
                '''
                Iterate (action, new state, reward, probability) items.

                Returns:
                    Generator[ItemsView[int, dict], None, None]: Generator of (action, new state, reward, probability) items.
                '''

                yield from self.map.items()

        self.grid = np.array([[Block(h, w) for w in range(width)] for h in range(height)])
        self.width, self.height = width, height

        self.terminalStates = np.zeros_like(self.grid, dtype = bool)

    def addTerminalState(self, y: int, x: int) -> None:
        '''
        Add new terminal state.

        Keyword Arguments:
            y (int): Y-coordinate of new terminal state.
            x (int): X-coordinate of new terminal state.
        '''

        self.terminalStates[y, x] = True
        self.grid[y, x].setMap({a: {((y, x), 0): 1} for a in range(4)})

    def isTerminal(self, state: Tuple[int, int]) -> bool:
        '''
        Checks if state is terminal state.

        Keyword Arguments:
            state (Tuple[int, int]): State to check.

        Returns:
            bool: True if state is terminal state, False otherwise.
        '''

        return self.terminalStates[state]
    
    def __setitem__(self, key: Tuple[int, int], value: dict) -> None:
        '''
        Set special map for given state.

        Keyword Arguments:
            key (Tuple[int, int]): State to change map.
            value (dict): Special map to set.
        '''
        
        self.grid[key].setMap(value)

    def __getitem__(self, key: Tuple[int, int]) -> object:
        '''
        get Block object of given state.

        Keyword Arguments:
            key (Tuple[int, int]): State to get block.
        
        Returns:
            object: Block object of given state.
        '''
        
        return self.grid[key]
    
    def __iter__(self) -> Generator[Tuple[int, int, object], None, None]:
        '''
        Iterate blocks of grid.

        Returns:
            Generator[Tuple[int, int, object], None, None]: Generator of grid coordinates and Block objects.
        '''
        
        for x in range(self.width):
            for y in range(self.height):
                yield y, x, self.grid[y, x]

    def plotStateValueFunction(self, values: np.ndarray) -> None:
        '''
        Plot the state value function.

        Keyword Arguments:
            values (np.ndarray): Array of state values.
        '''

        plt.figure(figsize = (10, 5))
        sns.heatmap(values, cmap = 'coolwarm', annot = True, fmt = '.1f', annot_kws = {'size': 16}, square = True)
        plt.title('State Value Function')
        plt.show()

    def plotActionValueFunction(self, qValues: np.ndarray) -> None:
        '''
        Plot the action value function.

        Keyword Arguments:
            qValues (np.ndarray): Array of action values.
        '''

        _, ax = plt.subplots(figsize = (self.width, self.height))
        cmap = sns.color_palette('vlag', as_cmap = True)
        norm = plt.Normalize(vmin = qValues.min(), vmax = qValues.max())

        for i in range(self.height):
            for j in range(self.width):
                Q = qValues[i, j]

                triUp = plt.Polygon([(j, self.height - i), (j + 0.5, self.height - i - 0.5), (j + 1, self.height - i)], facecolor = cmap(norm(Q[0])))
                triDown = plt.Polygon([(j + 1, self.height - i - 1), (j + 0.5, self.height - i - 0.5), (j, self.height - i - 1)], facecolor = cmap(norm(Q[1]))) 
                triLeft = plt.Polygon([(j, self.height - i), (j + 0.5, self.height - i - 0.5), (j, self.height - i - 1)], facecolor = cmap(norm(Q[2]))) 
                triRight = plt.Polygon([(j + 1, self.height - i), (j + 0.5, self.height - i - 0.5), (j + 1, self.height - i - 1)], facecolor = cmap(norm(Q[3]))) 
                
                ax.add_patch(triUp)
                ax.add_patch(triDown)
                ax.add_patch(triLeft)
                ax.add_patch(triRight)
                
                ax.text(j + 0.5, self.height - i - 0.17, f'{Q[0]:.2f}', ha = 'center', va = 'center', fontsize = 6)
                ax.text(j + 0.5, self.height - i - 0.83, f'{Q[1]:.2f}', ha = 'center', va = 'center', fontsize = 6)
                ax.text(j + 0.17, self.height - i - 0.5, f'{Q[2]:.2f}', ha = 'center', va = 'center', fontsize = 6)
                ax.text(j + 0.83, self.height - i - 0.5, f'{Q[3]:.2f}', ha = 'center', va = 'center', fontsize = 6)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.colorbar(plt.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax)
        
        plt.tight_layout()
        plt.show()