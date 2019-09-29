# Differentiable Cloth Simulation for Inverse Problems

Junbang Liang, Ming C. Lin, Vladlen Koltun. NeurIPS 2019.

[Project Page](https://gamma.umd.edu/researchdirections/virtualtryon/differentiablecloth)

### Requirements
- Python 3.6.4
- [PyTorch](https://pytorch.org/) tested on version 1.0.1
- CUDA 9.2.148
- cuDNN 7.0.5
- Pybind11
- BLAS
- Boost
- freeglut
- gfortran
- LAPACK
- libpng

### Build

1. Build the dependencies:
```
cd ${root}/arcsim/dependencies; make
```

2. Setup python libraries:
```
cd ${root}; make
```

To use the simulator:
```
import torch
import arcsim
```

For APIs of the simulator, please refer to pybind/bind.cpp.

### Demo

As the first step, link the simulation-related directories to the demo path:
```
cd demo
ln -s ../arcsim/conf .
ln -s ../arcsim/meshes .
ln -s ../arcsim/materials .
```

#### Simple Optimization of Gravity Force
The goal is to optimize the gravity force so that the center of mass of the cloth has the largest z coordinate after 1 second.
Execution command:
```
python demo_gravity.py
```

#### Collision Stress Test
This is the ablation study mentioned in the paper.
To see the backward timing, first uncomment Line 17 of pysim/collision_py.py.
Execution command:
```
python demo_collision.py ${log_dir}
```

#### Material Parameter Optimization
The goal is to optimize the density, stretching and bending parameters of the cloth so that the cloth behaves the same as observed.
Execution command:
```
python demo_wind.py ${log_dir} ${observed_data_dir} ${gt_material_file}
```

#### Motion Control
The goal is to find appropriate control forces (expressed as additional velocity very step) of the four corners of the cloth so that the cloth can be lifted and dropped down to the given basket, while avoiding the obstacles on the way.
Execution command:
```
python demo_throw.py ${log_dir}
```
There is another approach of this task which is to use a simple neural network to decide the forces:
```
python demo_embed.py ${log_dir}
```

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{liang2019differentiable,
  title={Differentiable Cloth Simulation for Inverse Problems},
  author = {Junbang Liang and Ming C. Lin and Vladlen Koltun},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

