**Model Predictive Control (MPC)**

The steps for the project:

* Choose state, input, dynamics, constraints and implement MPC
* Tune parameters
* Test solution on the simulator

## 1. MPC
Implementation of MPC controller is very similar to one from mpc_to_line quizz. It's implemented in MPC.cpp and MPC.h files.

At each moment we have a set of waypoints, we fit them to a third degree polynom, then we try to find required steering angle and acceleration to follow the waypoints as close as possible. To find them we use a set of constraints (including dynamic model) and optimization algorithm.

State vector consists of six elements - (x, y) - position of the vehicle, psi - orientation, v - speed, cte - cross track error, epsi - orientation error. Position and orientation are converted first from global coordinate system to vehicle's one.

Actuators is a vector of two elements - delta - steering angle, a - acceleration.

The cost of our model to optimize consists of cross track error (most important part, has largest weight), orientation error, differense between current and reference speed, steering angle, acceleration, steering angle and acceleration changes between steps (see comments in MPC.cpp).

Constraints contain dynamic model to update the vehicle's state from step to step.

## 2. Tuning Parameters

Parameters which were tuned: dt - timestep length, N - number of steps, ref_v - reference velocity. During implementation, debugging and finding better weights for cost parts I used slower vehicle and less number of states: ref_v = 20, dt = 0.2, N = 10.
Timestep length should be proportional to the vehicle's speed. So when MPC was implemented and velocuty was incresed to 40 miles/hour I decreased dt to 0.1. But after adding 100ms latency I increased dt to 0.15 to have it larger then latency value.
The result values: ref_v = 40, dt = 0.15, N = 20.

## 3. Testing
Tuning parameters and testing on the simulator were done simultaneously. I used simulator to find errors, find better cost weights and parameters. As the result car drives successfully the whole track several times.

