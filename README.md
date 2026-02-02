# Scientific and Physics-Informed Machine Learning in MATLAB &reg;

![plot of heat equation solution and physics informed neural network approximation](./ref/heat.png)

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=matlab-deep-learning/SciML-and-Physics-Informed-Machine-Learning-Examples)

This repository collates a number of examples demonstrating Scientific Machine Learning (SciML) and Physics-Informed Machine Learning.

Scientific Machine Learning is the application of Artificial Intelligence (AI) methods to accelerate scientific and engineering discoveries. SciML methods can incorporate domain specific knowledge, such as mathematical models of a physical system, with data-driven methods, such as training neural networks. Some characteristic SciML methods include:

* [Physics-Informed Neural Networks](https://doi.org/10.1016/j.jcp.2018.10.045) (PINNs) train a neural network to solve a differential equation by incorporating the differential equation in the loss function of the neural network training, utilizing the [automatic differentiation](https://uk.mathworks.com/help/deeplearning/ug/deep-learning-with-automatic-differentiation-in-matlab.html) (AD) framework to compute the required derivatives.
* [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (Neural ODEs) incorporate solving an ordinary differential equation as a [layer of a neural network](https://uk.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.neuralodelayer.html), allowing parameters that configure the ODE function to be [learnt via gradient descent](https://uk.mathworks.com/help/deeplearning/ref/trainnet.html).
* [Neural Operators](https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf) use neural networks to learn mappings between function spaces, for example a neural operator may map the boundary condition of a PDE to the corresponding solution of the PDE with this boundary condition.
* Graph Neural Networks (GNNs) incorporate graph structures in the data into the neural network architecture itself such that information can flow between connected vertices of the graph. Graph data structures often underly the numerical solutions of PDEs, such as the meshes in the [finite-element method](https://uk.mathworks.com/help/pde/ug/basics-of-the-finite-element-method.html).

## Getting Started

Download or clone this repository and explore the examples in each sub-directory using MATLAB&reg;.

## MathWorks&reg; Products

Requires [MATLAB&reg;](https://uk.mathworks.com/products/matlab.html).
* [Deep Learning Toolbox&trade;](https://uk.mathworks.com/products/deep-learning.html)
* [Partial Differential Equation Toolbox&trade;](https://uk.mathworks.com/products/pde.html)
  * For [Graph Neural Networks for Heat Transfer](./graph-neural-network-for-heat-transfer-problem/), [Inverse Problems Using Physics-Informed Neural Networks](./inverse-problems-using-physics-informed-neural-networks/), [Physics-Informed Neural Networks for Heat Transfer](./physics-informed-neural-networks-for-heat-transfer/).

# Physics-Informed Neural Network (PINNs) examples

[Physics-Informed Neural Networks](https://uk.mathworks.com/discovery/physics-informed-neural-networks.html) are neural networks that incorporate a differential equation in the loss function to encourage the neural network to approximate the solution of a PDE, or to solve an inverse problem such as identifying terms of the governing PDE given data samples of the solution. Automatic differentiation via [`dlarray`](https://uk.mathworks.com/help/deeplearning/ref/dlarray.html) makes it easy to compute the derivatives terms in the PDE via [`dlgradient`](https://uk.mathworks.com/help/deeplearning/ref/dlarray.dlgradient.html) for derivatives of scalar quantities, [`dljacobian`](https://uk.mathworks.com/help/deeplearning/ref/dlarray.dljacobian.html) for computing Jacobians, and [`dllaplacian`](https://uk.mathworks.com/help/deeplearning/ref/dlarray.dllaplacian.html), [`dldivergence`](https://uk.mathworks.com/help/deeplearning/ref/dlarray.dldivergence.html) for computing Laplacians and divergences respectively.

Explore the following examples on PINNs.

* [Physics-Informed Neural Networks for Mass Spring System](./physics-informed-neural-networks-for-mass-spring-system/)
* [Physics-Informed Neural Networks for Heat Transfer](./physics-informed-neural-networks-for-heat-transfer/)
* [Physics-Informed Neural Networks for Cavity Flow](./physics-informed-neural-networks-for-steady-cavity-flow/)
* [Inverse Problems using PINNs](./inverse-problems-using-physics-informed-neural-networks/)
* [Solve PDE Using Physics-Informed Neural Network](https://uk.mathworks.com/help/deeplearning/ug/solve-partial-differential-equations-with-lbfgs-method-and-deep-learning.html)
* [Solve Poisson Equation on Unit Disk Using Physics-Informed Neural Networks](https://uk.mathworks.com/help/pde/ug/solve-poisson-equation-on-unit-disk-using-pinn.html)
* [Solve Complex Partial Differential Equation Using Physics Informed Neural Network](./solve-complex-pde-using-pinn-example)

The following video content for PINNs is also available:

* [Using Physics-Informed Machine Learning to Improvie predictive Model Accuracy with Dr. Sam Raymond](https://uk.mathworks.com/company/user_stories/case-studies/using-physics-informed-machine-learning-to-improve-predictive-model-accuracy.html)
* [Physics-Informed Neural Networks - Podcast hosted by Jousef Murad with Conor Daly](https://youtu.be/eKzHKGVIZMk?feature=shared)
* [Physics-Informed Neural Networks with MATLAB - Live Coding Session hosted by Jousef Murad with Conor Daly](https://www.youtube.com/live/7ZdALJ2bIKA?feature=shared)
* [Physics-Informed Neural Networks with MATLAB - Deep Dive Session hosted by Jousef Murad with Conor Daly](https://youtu.be/RTR_RklvAUQ?feature=shared)

# Neural Differential Equation examples

Neural ordinary differential equations incorporate solving an ODE as a fundamental operation in a model, for example as a layer in a neural network such as [`neuralODELayer`](https://uk.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.neuralodelayer.html), or an ODE solver like [`dlode45`](https://uk.mathworks.com/help/deeplearning/ref/dlarray.dlode45.html) that integrates with the automatic differentiation framework such as the [`dlarray`](https://uk.mathworks.com/help/deeplearning/ref/dlarray.html) data type. Neural ODE solvers can be used in methods relevant to engineering such as [neural state space](https://uk.mathworks.com/help/ident/ug/what-are-neural-state-space-models.html) models that can be trained with [`idNeuralStateSpace`](https://uk.mathworks.com/help/ident/ref/idneuralstatespace.html) and [`nlssest`](https://uk.mathworks.com/help/ident/ref/nlssest.html) from [System Identification Toolbox&trade;](https://www.mathworks.com/products/sysid.html).

Explore the following examples on neural ODEs.

* [Universal Differential Equations](./universal-differential-equations/)
* [Dynamical System Modeling Using Neural ODE](https://uk.mathworks.com/help/deeplearning/ug/dynamical-system-modeling-using-neural-ode.html)
* [Train Latent ODE Network with Irregularly Sampled Time-Series Data](https://uk.mathworks.com/help/deeplearning/ug/train-latent-ode-network-with-irregularly-sampled-time-series-data.html)

# Neural Operator examples

Neural operators are typically used to learn mappings between infinite dimensional function spaces. Many PDE problems can be phrased in terms of operators. For example the parameterised differential operator $L_a$ defined by $L_a u := \nabla \cdot \left(a \nabla u\right)$ can be used to specify a Poisson problem $L_a u  = \nabla \cdot \left(a \nabla u \right) = f$ on a domain $\Omega$ as an operator problem. Given appropriate sets of functions $f \in \mathcal{V}$ and $a \in \mathcal{W}$, let $\mathcal{U}$ denote the set of solutions $u$ satisfying Dirichlet boundary condition $u = 0$ on $\partial \Omega$, and $L_a u = f$ on $\Omega$ for some $f \in \mathcal{V}$ , $a \in\mathcal{W}$. The solution operator $G:\mathcal{V} \times \mathcal{W} \rightarrow \mathcal{U}$ is defined as $G(f,a)= u$ such that $L_a u = f$.  Neural operator methods train neural networks $G_\theta$ to approximate the operator $G$. A trained neural operator $G_\theta$ can be used to approximate the solution $u$ to $L_a u = f$ by evaluating $G_\theta (f,a)$ for any $f \in \mathcal{V}, a \in \mathcal{W}$

* [Fourier Neural Operator](./fourier-neural-operator/)
* [Battery Module Cooling Analysis with Fourier Neural Operator](./battery-module-cooling-analysis-with-fourier-neural-operator)

# Graph Neural Network (GNNs) examples

Graph neural networks are neural network architectures designed to operate naturally on graph data $G = (E,V)$, where $V = \{v_1, \ldots, v_n\}$ is a set of $n$ vertices, and $E$ is a set of edges $e = (v_i,v_j)$ specifying that vertices $v_i$ and $v_j$ are connected. Both the vertices and edges may have associated features. Graph neural networks use natural operations for graph data such as graph convolutions which generalise a standard 2d discrete convolution. 

Explore the following examples on GNNs.

* [Graph Neural Networks for Heat Transfer](./graph-neural-network-for-heat-transfer-problem/)
* [Multivariate Time Series Anomaly Detection Using Graph Neural Network](https://uk.mathworks.com/help/deeplearning/ug/multivariate-time-series-anomaly-detection-using-graph-neural-network.html)
* [Node Classification Using Graph Convolutional Network](https://uk.mathworks.com/help/deeplearning/ug/node-classification-using-graph-convolutional-network.html)

# Hamiltonian Neural Network examples

The Hamiltonian neural network method trains a neural network to approximate the Hamiltonian of a mechanical system, as specified in [Hamiltonian formulation of mechanics](https://en.wikipedia.org/wiki/Hamiltonian_mechanics). The Hamiltonian formulation of mechanics specifies a mechanical system in terms of generalized coordinates $(q,p)$ where $q$ is a vector representation of the position of a point mass, and $q$ is a vector representation of the momentum. Hamiltonian mechanics specifies that the evolution of $p(t)$ and $q(t)$ in time can be specified by the ODE system $\frac{\mathrm{d}q}{\mathrm{d}t} = \frac{\partial H}{\partial p}$, $\frac{\mathrm{d}p}{\mathrm{d}t} = - \frac{\partial H}{\partial q}$ where $H(p,q,t)$ is called the Hamiltonian. By approximating $H$ by a neural network $H_\theta$ it is possible to compute $\frac{\partial H_\theta}{\partial p}, \frac{\partial H_\theta}{\partial q}$ and impose a loss based on the ODE system above, similar to the method of PINNs. 

* [Hamiltonian Neural Network](./hamiltonian-neural-network/)

## License
The license is available in the [license.txt](./license.txt) file in this GitHub repository.

## Community Support
[MATLAB Central](https://www.mathworks.com/matlabcentral)

Copyright 2024-2026 The MathWorks, Inc.
