# pyMAC

This code is an implementation designed to enable the usage of the code for MASS [Lee et al, 2019], Generative GaitNet [Park et al, 2022], and Bidirectional GaitNet [Park et al, 2023] solely through Python libraries without the need for C++.

We checked this code works in Python 3.8, ray(rllib) 2.0.1 and Cluster Server (64 CPUs (128 threads) and 1 GPU (RTX 3090) per node).


## Installation

1. Create Vritual Environment
```bash
python3.8 -m venv pyMAC
source pyMAC/bin/activate
```

2. DartPy installtion from source 

```bash
# Install dependencies following https://dartsim.github.io/install_dartpy_on_ubuntu.html
(pyMAC) git clone https://github.com/dartsim/dart.git
(pyMAC) cd dart
(pyMAC) cp {project_directory}/cpp_files/* python/dartpy/dynamics/
(pyMAC) git checkout tags/v6.11.1
(pyMAC) mkdir build
(pyMAC) cd build
(pyMAC) cmake .. -DCMAKE_BUILD_TYPE=Release -DDART_BUILD_DARTPY=ON
(pyMAC) make -j4 dartpy
(pyMAC) make install-dartpy
```

3. Install other libraries

```bash
pip install --upgrade pip

# Simulation and viwer libraries
pip3 install PyOpenGL imgui glfw numpy numba gym bvh numpy-quaternion

# DeepRL library 
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
## For GaitServer
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip3 install ray==2.0.1 
pip3 install ray[rllib] 
pip3 install ray[default]

# (Optional) if not working with previous installation
pip install gym==0.21.0 ## Problemn related to np.bool 
pip3 install "pydantic<2"
```
## Render 

```bash
cd {project folder}/
python3 viewer.py
```

## Learning 

```bash
cd {project folder}/
python3 train.py --config={configuration}
```

## Update Log
- [x] Implementation of imitation learning (deep mimic / scadiver) torque version (2023.03.05 (Verified))

- [x] Implementation of imitation learning muscle version 

- [ ] Attach video2motion frameworks

- [ ] Fast rendering of muscle 

- [ ] Fast rendering of high-resolution mesh

- [ ] Support MacOS (Converting glfw to glut)
# MuscleRetargetting
