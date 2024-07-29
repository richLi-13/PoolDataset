# This is a project that generates a pool dataset in pairs of (state, goal).
The file goal.py generates the goal dataset (directly, indirectly, bank shot).
The file state.py generates the state dataset.
The file check_feasibilty.py randomly chooses a goal and a state, and decide whether the goal can be achieved under the state. 
    If so, add the pair to feasible_dataset.txt. Otherwise, discard it.
The file state.py puts everything together and takes all possible parameters as the input.
