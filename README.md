# Reproducing Our Results
Import the model into main.py and specify the desired output folder for the training as the
"MODEL" constant. If you plan on using cuda to train then you need to switch the Coach import from AsyncCoach to Coach,
but cuda can also be disabled by editing the model's NNet.py args. 
Models are located in connect4/. 

# The Solver
We are using a solver created by Pascal Pons, more information and links to the source
can be found at [step by step tutorial to build a perfect Connect 4 AI](http://blog.gamesolver.org).

# The Environment