# solve_sycamore
This repo contains data, contraction code, and contraction order for the paper
[''Solving the sampling problem of the Sycamore quantum supremacy circuits''](https://arxiv.org/abs/2111.03011)

We provide demo code for reproducing the results in the paper, if you have enough GPUs :)



## Requirements

1. `pytorch` version greater than 1.7.0
2. A GPU with 32G memory or larger

## Usage
* Unzip the `src/scheme.tar.gz` to get the contraction scheme file, 
* Run the demo code to obtain a result of one complete subtask using

```python
python src/demo.py -cuda 0 -get_time
```
If the running time is too long, add argument `-subtask_num 10` to run 10 out of 64 head subroutines and 10 out of 128 tail subroutines. The arguments `task_start`, `task_end`, and `task_num` are used to control the overall number of subtasks in one run. 

The samples obtained in the paper are stored in  `samples.tar.gz`, which contains $2^{20}$ bitstrings.

Notice that the first 8 edges in `contraction_scheme['slicing_edges_loop']` and their companion edges compose the drilling holes in the article.
