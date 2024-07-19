This directory contains all code required to reproduce the results from [Math_4250_Project_2.pdf](Math_4250_Project_2.pdf).

There are 6 scripts in this repository. Each can be run with the command `python fileName.py`, where `fileName` is the name of the file you want to run. `GridWorld.py` is effectively a library, implementing the class required for the other files to execute their algorithms, so running it will have no output. `value.py` will output $5 \times 5$ arrays of floats, representing value functions, and the rest of the files will output $5 \times 5$ arrays of arrows, representing policies. Each file will display its output visually with `matplotlib` and `seaborn`, but will not save it automatically. If you wish to save the figures as they are displayed, you will have to do so manually.

Sample value function:
```
[[ 2.2  4.7  2.1  1.3  1.8]
 [ 1.1  1.8  1.2  0.7  0.6]
 [ 0.2  0.5  0.4  0.1 -0.2]
 [-0.5 -0.3 -0.3 -0.4 -0.7]
 [-1.1 -0.8 -0.8 -0.9 -1.2]]
```

Sample policy:
```
[['→' '↑' '←' '←' '↑']
 ['↑' '↑' '↑' '←' '←']
 ['↑' '↑' '↑' '←' '←']
 ['↑' '↑' '↑' '←' '↑']
 ['↑' '↑' '↑' '↑' '↑']]
```
