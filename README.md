# Install
```
python3 -m pip install torch
```

# Launch file
First launch the solution_stack.launch file to prepare all the nodes.

```
roslaunch solution_prep solution_stack.launch
```

# Conduct experiments
1. Use srv to conduct steps of experiments.

2. Or use orchestrator node to conduct the whole experiment of solution preparation at a target pH.
```
rosrun solution_prep orchestrator.py
```

Avaiable parameters:
```
self.V0_ml = float(rospy.get_param("~current_water_volume_ml", 20))  # Initial volume of pure water(mL) in target beaker
self.target_ph = float(rospy.get_param("~target_ph", 4.00))  # target pH
self.c_HCl = float(rospy.get_param("~c_HCl", 0.01585))  # source HCl mol/L
self.ph_tol = float(rospy.get_param("~ph_tolerance", 0.02))  # permitted pH error
self.max_loops = int(rospy.get_param("~max_loops", 6))  # maximum loops
```
