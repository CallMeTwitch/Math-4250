This directory contains all code required to reproduce the results from [Math_4250_Project_3.pdf](Math_4250_Project_3.pdf).

There are 4 scripts in this repository. Each can be run with the command `python fileName.py`, where `fileName` is the name of the file you want to run. `GridWorld.py` is effectively a library, implementing the class required for the other files to execute their algorithms, so running it will have no output. `value.py` will output $7 \times 7$ arrays of floats, representing value functions. `policy.py` will output $5 \times 5$ arrays of arrows, representing policies. `adversarial.py` will only display figures, no outputs. Each file will display its output visually with `matplotlib` and `seaborn`, as well as textually (where possible), but will not save automatically. If you wish to save the figures as they are displayed, you will have to do so manually.

Sample value function:
```
[[ 0.   0.   0.   0.1  0.2  0.5 -0. ]
 [-0.   0.   0.   0.1  0.1  0.3  0.5]
 [-0.  -0.   0.   0.   0.1  0.1  0.2]
 [-0.1 -0.1 -0.  -0.   0.   0.1  0.1]
 [-0.2 -0.1 -0.1 -0.   0.   0.   0. ]
 [-0.5 -0.3 -0.1 -0.1 -0.   0.   0. ]
 [-0.  -0.5 -0.2 -0.1 -0.  -0.   0. ]]
```

Sample policy:
```
[['↑' '←' '←' '→' '↑']
 ['↑' '↑' '↑' '↑' '↑']
 ['↑' '↑' '↑' '↑' '↑']
 ['→' '→' '↑' '←' '←']
 ['↑' '↑' '↑' '↑' '↑']]
```
