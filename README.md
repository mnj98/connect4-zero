# Reproducing Our Results
Import the model into main.py and specify the desired output folder for the training as the
MODEL constant. If you plan on using cuda to train then you need to switch the Coach import from AsyncCoach to Coach,
but cuda can also be disabled by editing the model's NNet.py args. 
Models are located in connect4/.
Once you've set the import to the desired model in main.py, run main.py from the project's root directory 
and the checkpoints will be stored in saved_checkpoints/MODEL.

# The Solver
We are using a solver created by Pascal Pons, more information and links to the source
can be found at [step by step tutorial to build a perfect Connect 4 AI](http://blog.gamesolver.org).

# The Environment
We are using an enviroment we found on by Suragnair on [Github](https://github.com/suragnair/alpha-zero-general).
We added tweaks to allow for different draw rewards/penalties during training, trimming old training example files, the
ability to rebase a model back to it's best model after X iterations fail to beat the best model, the ability to
stop and resume training while keeping the optimizer's parameters, Asynchronous Self-Play and Pitting, and
code to evaluate our models against the solver, and the ability to factor draws into the Pitting results.
