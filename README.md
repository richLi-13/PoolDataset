# This is the dissertation project that generates a pool dataset in pairs of (state, goal) and performs evaluation on it.

## Dataset Generation

The file goal.py generates the goal dataset (directly, indirectly, bank shot).

The file state.py generates the state dataset.

The file check_feasibilty.py randomly chooses a goal and a state, and decide whether the goal can be achieved under the state. If so, add the pair to feasible_dataset.txt. Otherwise, discard it.

The file script.py puts everything together and takes all possible parameters as the input. By running the "bigscript" function in script.py, a new dataset will be generated to the "feasibility_dataset_new.txt" file.

## Dataset Evaluation

The file pool.py and utils.py are the preparation of the evaluation system that uses PoolTool.

The file pool_solver.py contains the Bayesian optimization, the reward function, and the function mainly used for evaluation: the get_shot function.

The files multiprocessor.py and multiprocessor_non_zero.py use the evaluate_one function in script.py that performs the evaluation. By running these two files, each data point in "sample_indices_500.txt" will be evaluated and obtain a success rate, or whether it is successful or not.

By changing the SEARCH_STEPS in pool_solver.py and itr_times and output_file in multiprocessor_non_zero.py, the given sample files are generated.
