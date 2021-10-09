# DeLaN Metric-Ik planner in C++ #
## Dependencies ##
Libraries **mentioned in bold** Required for current version
* **libtorch** (pytorch c++ api )
* **nlopt** (optimization library in c++)
* **matplotlib** (to visualize graphs)
* **eigen3**  (for  matrix enhancement) 

* Current Optimizer is SLSQP with non-linear equalities and 2 linear inequalities

## Timing comparision ##
| Ik-Algorithm       | One-step planning Time | 
| ------------- |:-------------:| 
| **DeLaN Metric-IK planner (c++)**   | < 1 ms |
| off the shelve Ik ( ROS-KDL) planner | < 0.50 ms  |

