## Agile flight control using model predictive control and neural network
* In this project, I trained a neural net imitating a MPC to control a quacotper fly through a moving gate
* The project is based on previous work [1], [2]. In that work, author used traversal time as a high level decision variable (z) to improve MPC and used a neural network to learning traversal time from observation.
* In this work, I used a neural network to learn control output (thrust and 3 body rate) from observation. I run step-based simulation to collect data which include observation states and respective control output of MPC, I collected about 20000 samples and saved as a dataset for supervised learning. I built a simple neural net to train collected dataset. The neural network successfully learning to regress control values with low MSE error.
* The project baseline is based on [high-mpc](https://github.com/uzh-rpg/high_mpc)

### Update
* [03/02/2023] Initial commit
* Project is updating and reorganizing
### Simulation

![alt text](https://github.com/phuongboi/agile-flight-control-using-MPC-and-neural-net/blob/main/high_MPC/MPC/saved/output.gif)

### How to use
* Run `pip install -e` to install packages
* Check out file `run_supervise.py` to start the pipeline:
* STEP 1 : Collecting data: `data_collect = True`
*  STEP 2 : Training model: `training = True`
*  STEP 3 : Running experiment: `run = True`

### References
* [1] Song, Yunlong, and Davide Scaramuzza. "Policy search for model predictive control with application to agile drone flight." IEEE Transactions on Robotics 38.4 (2022): 2114-2130.
* [2] Song, Yunlong, and Davide Scaramuzza. "Learning high-level policies for model predictive control." 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.
* [3] https://github.com/uzh-rpg/high_mpc
