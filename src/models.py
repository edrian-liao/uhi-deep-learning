import math
import torch
import gpytorch
import numpy as np

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import Kernel, MaternKernel 

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ZeroMean()
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.LinearMean() # use this instead to 
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NonStationaryKernel(Kernel):
    """
    A non-stationary kernel that applies a parametric signal variance
    on top of a base stationary kernel, with multiple RBF widths.
    """
    def __init__(self, base_kernel=None, num_rbf_centers=100, input_dim=2):
        super(NonStationaryKernel, self).__init__()
        self.base_kernel = base_kernel

        n = int(np.sqrt(num_rbf_centers))
    
        # Define grid parameters
        grid_x = torch.linspace(0, 1.0, n)  # x-coordinates from 0.1 to 0.9 with step 0.1
        grid_y = torch.linspace(0, 1.0, n)  # y-coordinates from 0.1 to 0.9 with step 0.1

        # Generate all (x, y) coordinate pairs for the grid
        grid_points = torch.cartesian_prod(grid_x, grid_y)

        # Assign to self.rbf_centers
        self.register_buffer("rbf_centers", grid_points)

        # Define coefficients (c_k in g(x))
        self.register_parameter(
            name="coefficients",
            parameter=torch.nn.Parameter(torch.randn(num_rbf_centers))
        )

        # Initialize RBF widths constrained to (0, ∞) via softplus
        self.register_parameter(
            name="raw_rbf_widths",
            parameter=torch.nn.Parameter(torch.ones(num_rbf_centers))
        )

        # register the constraint
        self.register_constraint("raw_rbf_widths", gpytorch.constraints.Positive())

    # @property
    def rbf_widths(self):
        return self.raw_rbf_widths_constraint.transform(self.raw_rbf_widths)

    def forward(self, x1, x2, **params):
        g_x1 = self._compute_signal_variance(x1)
        g_x2 = self._compute_signal_variance(x2)
        base_k = self.base_kernel(x1, x2, **params)
        return g_x1.unsqueeze(-1) * g_x2.unsqueeze(-2) * base_k

    def _compute_signal_variance(self, x):
        distances = torch.cdist(x, self.rbf_centers) ** 2
        rbf_values = torch.exp(-distances / self.rbf_widths().unsqueeze(0)) # ensure no division by zero
        g_x = torch.matmul(rbf_values, self.coefficients)
        return g_x

# GP Model
class NonStationaryGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_points=100):
        super(NonStationaryGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = NonStationaryKernel(
            base_kernel=MaternKernel(nu=1.5), num_rbf_centers=num_points, input_dim=train_x.shape[-1]
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)