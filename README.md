# PPO algorithm for Car Racing in OpenAI gym
This project implements the Proximal Policy Optimization (PPO) algorithm with PyTorch, applied to the [CarRacing-v2](https://gymnasium.farama.org/environments/box2d/car_racing/) Box2D environment from the OpenAI Gym library.
## DEMO
![](https://github.com/EricChen0104/PPO_Car_Racing_OpenAI_gym/blob/master/plots/test_episode.gif)

### To run the code:
<br/> *cd to the **gym_car_racing** file in terminal. then run this:*
```Shell
python3.10 main_car_racing.py
```

## What is OpenAI Gym?
**OpenAI gym** is a research environment for *reinforcement learning(RL)* which was created by OpenAI. It prevent a simulation environment to make the researcher design, train or even test any of the RL algorithm more convenient.<br /> <br />
Gym supports many kinds of classical control problem. For instance, Rotary Inverted Pendulum, Discrete Action Space(like CartPole, MountainCar). Gym even includes high level vision and continuous control mission(like Atari and MuJoCo).<br /> <br />
The main idea of Gym is to emphasize the simplicity and scalability. It's API is extremely intuitive, it contains *four* main steps: initialize environment, reset, interaction(step) and render. Makes user focus on algorithm's develop and improvment since the developers do not need to spent time on handling the environment detail.<br /> <br />

## Why do this project?
Since OpenAI Gym provides well-resourced environment, it is not only be used in research. It can also be a fantastic environment to people which just started study RL. <br/> <br/>
However, in this project I use the **CarRacing_v2** environment of gym. Using the Python language with Pytorch to create **PPO algorithm with no library** for the reinforcement learning. <br/><br/>
Hope this project can make more people which is new to reinforcement learning some inspiration and provide a simple reference.

## What is PPO algorithm?
In short, PPO (Proximal Policy Optimization) is a reinforcemnet learning structure base on **Actor-Critic structure**. Which was performed by OpenAI. PPO is build by two main network: **Actor** decides action policy, **Critic** evaluates the current policy.  

### Why use PPO in Car Racing?
In OpenAI Gym's Car Racing environment, the action space is continuous. Which means traditional discrete control algorithm (like DQN) is not suitable. <br/><br/>
PPO can stably learn continuous control policy by the Actor-Critic structure. Although PPO has a well performance in high demention observe and stochatic environment, it is often be the algorithm for mission as CarRacing. <br/>
