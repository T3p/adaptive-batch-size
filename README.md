# adaptive-batch-size

This is the implementation of the safe, adaptive policy gradient methods described in the paper:

**Papini, Matteo, Matteo Pirotta, and Marcello Restelli. "Adaptive Batch Size for Safe Policy Gradients." Advances in Neural Information Processing Systems. 2017,**

which is available [here](http://papers.nips.cc/paper/6950-adaptive-batch-size-for-safe-policy-gradients).

In short, the paper describes some methods to adapt the **step size** (length of gradient updates) and the **batch size** (number of trajectories used to estimate the gradient) in order to achieve **safefy** (monotonic improvement) in **continous reinforcement learning** tasks.
More precisely, it describes variants of the **REINFORCE** and **G(PO)MDP** algorithms with **Gaussian policies** which require less tuning of meta-parameters and can guarantee monotonic improvement with high probability — at the cost of speed.

Here we provide the code to try the new algorithms on the **Linear Quadratic Gaussian Control** problem.


**Replicating the experiments**\
To replicate all the experiments of the paper, just clone the repository locally and run **exp_full.py**.
*Warning*: it may take a *long* time, so you may want to comment some lines of the script.

**Testing on other tasks**\
The main algorithm is in **adabatch.py**. To test the methods on other tasks, you can modify the LQG experiment in **exp_lqg1d.py**.
