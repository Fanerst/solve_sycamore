# solve_sycamore
Reproduce the random circuit sampling experiments of Sycamore quantum circuit

## Requirements

1. `pytorch` version greater than 1.7.0
2. A GPU with 32G memory or larger

## Usage
Firstly, unzip the `src/scheme.tar.gz` to get the contraction scheme file, then run
```python
python src/demo.py -cuda 0 -get_time
```
to get a result of one complete subtask. If the running time is too long, add argument `-subtask_num 10` to run 10 out of 64 for head and 10 out of 128 for tail subroutines. The arguments `task_start`, `task_end` and `task_num` are used to control the overall number of subtasks in one run. 

The sample file can be retrieved by extracting `samples.tar.gz`, which contains $2^{20}$ bitstrings of 53 qubits.

Notice that the first 8 edges in `contraction_scheme['slicing_edges_loop']` and their companion edges compose the drilling holes in the article.