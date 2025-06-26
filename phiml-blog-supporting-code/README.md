# Physics Informed Machine Learning Methods and Implementation supporting code
This collection provides examples demonstrating a variety of Physics-Informed Machine Learning (PhiML) techniques, applied to a simple pendulum system. The examples provided here are discussed in detail in the accompanying blog post "Physics-Informed Machine Learning: Methods and Implementation". 

![hippo](https://www.mathworks.com/help/examples/symbolic/win64/SimulateThePhysicsOfAPendulumsPeriodicSwingExample_10.gif)

The techniques described for the pendulum are categorized by their primary objectives:
- **Modeling complex systems from data** 
    1. **Neural Ordinary Differential Equation**: learns underlying dynamics $f$ in the differential equation $\frac{d \mathbf{x}}{d t} = f(\mathbf{x},\mathbf{u})$ using a neural network.
    2. **Neural State-Space**: learns both the system dynamics $f$ and the measurement function $g$ in the state-space model $\frac{d \mathbf{x}}{dt} = f(\mathbf{x},\mathbf{u}), \mathbf{y}=g(\mathbf{x},\mathbf{u})$ using neural networks. Not shown in this repository, but illustrated in the documentation example [Neural State-Space Model of Simple Pendulum System](https://www.mathworks.com/help/ident/ug/training-a-neural-state-space-model-for-a-simple-pendulum-system.html). 
    3. **Universal Differential Equation**: combines known dynamics $g$ with unknown dynamics $h$, for example $\frac{d\mathbf{x}}{dt} = f(\mathbf{x},\mathbf{u}) = g(\mathbf{x},\mathbf{u}) + h(\mathbf{x},\mathbf{u})$, learning the unknown part $h$ using a neural network. 
    4. **Hamiltonian Neural Network**: learns the system's Hamiltonian $\mathcal{H}$, and accounts for energy conservation by enforcing Hamilton's equations $\frac{dq}{dt} = \frac{\partial \mathcal{H}}{\partial p}, \frac{dp}{dt} = -\frac{\partial \mathcal{H}}{\partial q}$.
- **Discovering governing equations from data:** 
    1. **SINDy (Sparse Identification of Nonlinear Dynamics)**: learns a mathematical representation of the system dynamics $f$ by performing sparse regression on a library of candidate functions. 
- **Solving known ordinary and partial differential equations:** 
    1. **Physics-Informed Neural Networks**: learns the solution to a differential equation by embedding the governing equations directly into the neural network's loss function.  
    2. **Fourier Neural Operator**: learns the solution operator that maps input functions (e.g. forcing function) to the solution of a differential equation, utilizing Fourier transforms to parameterize the integral kernel and efficiently capture global dependencies.   

## Setup
Open the project file physics-informed-ml-blog-supporting-code.prj to correctly set the path.  

## MathWorks Products ([https://www.mathworks.com](https://www.mathworks.com/))
Requires MATLAB&reg; R2024a or newer
- [Deep Learning Toolbox&trade;](https://www.mathworks.com/products/deep-learning.html)

## License
The license is available in the [license.txt](license.txt) file in this Github repository.

## Community Support 
[MATLAB Central](https://www.mathworks.com/matlabcentral)

Copyright 2025 The MathWorks, Inc. 

