# Solve Complex Partial Differential Equation Using Physics Informed Neural Network

This example shows how to train a physics\-informed neural network to simulate the behavior of electrons inside a simplified 2\-D condensed matter system by solving the Schrödinger equation, a complex\-valued partial differential equation.

The code in this example is based on work by Hsu et al \[1,2\]. Periodic boundary conditions are enforced using techniques developed by Shavinger et al \[3\].

Example live script: 
- [SolveComplexPDEUsingPINNExample.m](./SolveComplexPDEUsingPINNExample.m)

Supporting files:
- [atomicPseudoPotential.m](./atomicPseudoPotential.m)
- [plotBandStructure.m](./plotBandStructure.m)
- [plotWaveFunction.m](./plotWaveFunction.m)
- [realSpaceSampling.m](./realSpaceSampling.m)
- [reciprocalSpaceSampling.m](./reciprocalSpaceSampling.m)

# MathWorks® Products

Requires MATLAB® release R2025a or newer.

This example uses:

- [Deep Learning Toolbox™](https://mathworks.com/products/deep-learning.html)

# License

The license is available in the [license.txt](https://github.com/matlab-deep-learning/SciML-and-Physics-Informed-Machine-Learning-Examples/blob/main/universal-differential-equations/license.txt) file in this GitHub repository.

# Community Support

[MATLAB Central](https://www.mathworks.com/matlabcentral)

*Copyright 2026 The MathWorks, Inc.*

# References
<a id="M_9aca"></a>
\[1\] Hsu, C., M. Mattheakis, G. R. Schleder, and D. T. Larson. "Equation\-driven Neural Networks for Periodic Quantum Systems". Machine Learning and the Physical Sciences Workshop, NeurOPS 2024. [https://ml4physicalsciences.github.io/2024/files/NeurIPS\_ML4PS\_2024\_165.pdf](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_165.pdf)

\[2\] [https://github.com/circee/blochnet](https://github.com/circee/blochnet) 

<a id="M_54cd"></a>
\[3\] Shaviner, G. G., H. Chandravamsi, S. Pisnoy, Z. Chen, and S. H. Frankel. "PINNs for Solving Unsteady Maxwell's Equations: Convergence Issues and Comparative Assessment with Compact Schemes". ArXiv. [https://arxiv.org/pdf/2504.12144](https://arxiv.org/pdf/2504.12144)