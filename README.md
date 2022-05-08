# 16.412 Final Project

## Installation

Run the following commands to install all the relevant packages/libraries.
```
pip3 install -r requirements.txt
sudo apt-get install python3-tk
```

Add the following to `~/.bashrc` to set the correct env variables:
```
export GAZEBO_MODEL_PATH=~/catkin_ws/src/final_proj_sim/models:/usr/share/gazebo-9/models
```

## Running the scripts

To run the planner without 3d simulations use this:
```
python3 scripts/planner_test.py
```

To run the planner **with** 3d simulations use this:
```
python3 scripts/sandbox.py
gazebo worlds/gen.world
```

