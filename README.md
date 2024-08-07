This repository contains the projects create for MATH 4250 (Reinforcement Learning) at MUN. These files are NOT to be used by anyone for any reason to cheat in 4250. 

All project files can be installed using the following:
```
git clone https://github.com/CallMeTwitch/Math-4250.git
```

Individual project instructions can be found in individual project README files.

## Project 1: K-Armed Bandit Problem
Abstract: This paper presents a novel probabilistic strategy for addressing the k-armed bandit problem, leveraging probability computations derived from observed rewards. Our method is benchmarked against standard algorithms such as greedy, epsilon-greedy, optimistic initialization, and gradient bandit strategies. Experimental results demonstrate that the proposed method achieves a ~5\% improvement in average accumulated reward (AAR) and ~10\% improvement in percent optimal action (POA) over most standard algorithms in stationary environments, and an increase over all standard algorithms in some non-stationary environments, making it a competitive alternative to existing methods.

[Accompanying Paper](<K-Armed Bandit Problem/Math_4250_Project_1.pdf>)

## Project 2: GridWorld
Abstract: This paper implements and analyzes a wide range of methods for solving Markov Decision Processes (MDPs) in a $5 \times 5$ GridWorld environment including explicit solutions, iterative methods, and Monte Carlo approaches. By applying these techniques to both stationary and dynamic variants of the environment, we gain insight into their relative strengths, limitations, and practical applications. This report presents our findings, comparing the performance and characteristics of different RL algorithms in solving the GridWorld problem. Through this analysis, we aim to deepen our understanding of core RL concepts, and contextualize their place in modern AI.

[Accompanying Paper](<GridWorld/Math_4250_Project_2.pdf>)

## Project 3: GridWorld Extended
Abstract: This paper extends the experiments and results conducted and derived in our previous GridWorld paper by implementing different algorithms (including SARSA, Q-Learning, Gradient Monte Carlo, and Semi-Gradient $\text{TD}(0)$) in more complex environments. In applying these methods to more complex variations of the GridWorld environment (including a custom adversarial 'predator-prey' setup) we hope to test the limits of these algorithms, and gain real insight into their individual strengths and weaknesses. This paper reports our findings, comparing the abilities of these algorithms in solving complex GridWorld problems. Through these experiements, we hope to extend the boundary of our knowledge in RL, and demonstrate the practical capabilities of these algorithms in the real world.

[Accompanying Paper](<GridWorldExtended/Math_4250_Project_3.pdf>)
