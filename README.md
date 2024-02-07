## Agile flight control using model predictive control and neural network
* In this project, I trained a neural net imitating a MPC to control a quadcopter fly through a moving gate
* The project is based on previous work [1], [2]. In that work, author used traversal time as a high level decision variable (z) to improve MPC and used a neural network to learning traversal time from observation.
* In this work, I used a neural network to learn control output (thrust and 3 body rate) from observation. I run step-based simulation to collect data which include observation states and respective control output of MPC, I collected about 20000 samples and saved as a dataset for supervised learning. I built a simple neural net to train collected dataset. The neural network successfully learning to regress control values reach MSE around 0.04 after 400 epoch.
* The project baseline is based on [high-mpc](https://github.com/uzh-rpg/high_mpc)

### Update
* [07/03/2024] I concated quadcopter state and pendulum state to make an input observation with size 18, model reach 0.01 MSE after 400 epoch, quadcopter can learning more accurate action under the guide of MPC and don't fly wildly in the end of each episode
* [03/02/2024] Initial commit
* Project is updating and reorganizing
### Simulation
#### Update result, 07/03/2024
![alt text](https://github.com/phuongboi/agile-flight-control-using-mpc-and-neural-net/blob/main/high_mpc/mpc/saved/output2.gif)
#### Init result, 03/02/2024
![alt text](https://github.com/phuongboi/agile-flight-control-using-mpc-and-neural-net/blob/main/high_mpc/mpc/saved/output.gif)

### How to use
* Run `pip install -e` to install packages
* Check out file `run_supervise.py` to start the pipeline:
* STEP 1 : Collecting data: `python run_supervise.py --data_collect`
*  STEP 2 : Training model: `run_supervise.py --training`, using file `high_mpc/Dataset/obs2_40000.npz`
*  STEP 3 : Running experiment: `python run_supervise.py --run`, using checkpoint `high_mpc/weights/obs2_40k/ep_400.pth`

### Reference
* [1] Song, Yunlong, and Davide Scaramuzza. "Policy search for model predictive control with application to agile drone flight." IEEE Transactions on Robotics 38.4 (2022): 2114-2130.
* [2] Song, Yunlong, and Davide Scaramuzza. "Learning high-level policies for model predictive control." 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.
* [3] https://github.com/uzh-rpg/high_mpc
