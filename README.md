## Domain Transfer for Reinforcement Learning Agents

Data Innovation Lab winter semester 2019

## Project Description

Supervised models have proven very effective when it comes to specialized tasks 
such as image classification and object detection. Real world systems are how ever
often much larger and more complicated and may consist of several machine learning
components working together. Such as system might be hard to optimize in a supervised
fashion as there might not be a known mapping between observations and actions.
Reinforcement learning agents on the other hand are more flexible and do not rely
on a dense labels and a differentiable loss in order to learn. Instead they adapt
their behavior to maximize a well defined reward function. In combination with deep learning
reinforcement agents have recently shown impressive performance complex tasks such as
controlling robots and playing video games.

Training reinforcement learning agents for real systems is in practice complicated
since in order to learn the agent needs to act with the environment. Most RL algorithm
are also very data hungry and might require millions of episodes before performing on
par with humans on the same task. Training an agent in the real physical environment is
therefore most often neither safe or feasible and most RL agents are trained and tested
in simulated environments before deployed into the real world. Additionally there is no
guarantee that an agent trained in the simulation will also perform well in the real physical
domain. Since the dynamics of the simulation deviates from the dynamics of the physical
world agents will most likely underperform or completely fail once deployed in the new domain.
In this project we will explore recent methods to generalize RL agents trained in one environment,
to also perform well when deployed into a new environment. The goal of the project is to develop a
method to train RL agents that can perform in a real physical environment even if they were only
trained on recorded and simulated data. The transferability of developed algorithm will be
benchmarked on public RL environments as well as on a simulation of the target domain.


## Timeline
TBD

## Related Blog Posts

Generalisation from Simulation: https://openai.com/blog/generalizing-from-simulation/

## Related Research Papers
| Paper | Summary |
|---|---|
|[Modular Vehicle Control for Transferring Semantic Information Between Weather Conditions Using GANs](https://arxiv.org/abs/1807.01001)|Transfer RL controller to new weather conditions|


## Links and resources

At PreciBake we use Python 3 and PyTorch for machine learning development. 
For a good introduction to python for data science I recommend the cs231 tutorial
http://cs231n.github.io/python-numpy-tutorial/

On the training server we use docker containers as training enviroments. 
You are not required to build your own docker containers, but it is good to be familiar
with the core concepts as you will need to start up a container in order to run PyTorch and train on the GPU.

A good introduction to docker is found here:
https://docs.docker.com/get-started/#test-docker-installation


### Pytorch Tutorials:

Pytorch tutorial: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

Pytorch minimal mnist: https://github.com/maitek/pytorch-examples/blob/master/mnist.py

Pytorch RL tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

### Reinforcement learning tutorials

Andrej Karpathys blog on Deep RL http://karpathy.github.io/2016/05/31/rl/

OpenAI Spinning up: https://spinningup.openai.com/en/latest/index.html

OpenAI Gym: https://gym.openai.com/

Berkeley Deep RL course: http://rail.eecs.berkeley.edu/deeprlcourse/



### Precibake Environment:

COMING SOON..

## Git Howto:

In order easily collaborate in the Gitlab repository I suggest the following workflow:

1. Create a new (feature) branch.
```
git checkout -b feature/my_branch
```
2. Commit and push all changes to the branch.
```
git add my_file.py
git commit -m "Some useful commit message here"
```
3. Once you want to merge your branch with master. Create a merge request in Gitlab.
4. Review changes and merge the merge request to master in Gitlab.

Pushing directly on Master branch has been disabled for now. Instead you need to create a merge request, and merge it through gitlab. This helps the master branch to stay clean, and reduces the risk of conflicts. By creating merge requests in gitlab you can easily see what has changed to your branch before merging. This way you can make sure you only merge what is necessary and not merge anything by accident.

Some tips:
- Merge your new features often (once a week at least), to avoid large merge conflicts.
- Rebase or merge your feature with master before making a merge request to avoid conflicts.
- You can use "git status" to check current branch, and unpushed commits
- You can use "git log" to see your last commit.
- Commit and push often
- Merge often to avoid conflicts

If you are new to git I suggest this short tutorial:

https://tutorialzine.com/2016/06/learn-git-in-30-minutes

Merging your own merge request should now be allowed in the Gitlab repo. Let me know if you have problems.

## GPU Server Howto

We have setup a GPU server where you can run your experiments if needed. 

You can login to the server using ssh in our office network:

```
ssh user@10.1.5.157 -p 8024
```

We have prepeared a docker image with pytorch and other common deep learning libraries.
To run your scripts you can start a docker container using a the following bash script located in the project folder.

```
sh start_docker.sh
```

If you are unfamiliar with the conecept of docker you can readup here:
https://docs.docker.com/get-started/#test-docker-installation

### Install additional python packages in docker
The docker container should have most of what you need. But if you need to install some additional python packages you can safely do so.

Using conda:
```
/opt/conda/bin/conda install sklearn
```
Using pip:
```
/opt/conda/bin/pip install sklearn --user
```
Note that installing things in docker container is NOT presistant, so you will need to install it every time you start the docker.

### Use TMUX to prevent your training to stop when closing the ssh session
When closing the ssh session your running scripts will be suspended. 
One way to prevent this is to start your docker inside TMUX.

To start a tmux session:

```
tmux
```

To attach to an exsisting tmux:

```
tmux attach
```

https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/


### Select GPU to train on
If you get an memory error while training it might be because one of the GPUs is occupied.

You can the status of the gpus using:
```
nvidia-smi
```

You can select the second gpu by using exporting CUDA_VISIBLE_DEVICES=GPU_ID variable:

```
export CUDA_VISIBLE_DEVICES=1
```
