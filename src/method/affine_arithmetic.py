import numpy as torch
import logging
import math

from method.interval_arithmetic import Interval, interval_hull, intersection
from method.method_plugin_abc import MethodPluginABC

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class AffineNN(MethodPluginABC):
    """
    AffineNN is a method plugin for performing affine arithmetic-based interval analysis 
    on neural networks. It supports interval tightening through learnable ReLU parameters 
    and gradient-based optimization.
    Attributes:
        module (nn.Module): The neural network module to analyze.
        epsilon (float): The interval radius.
        optimize_bounds (bool): Flag indicating whether to optimize bounds using gradient-based methods.
        gradient_iter (int): Number of gradient iterations for optimizing bounds (required if optimize_bounds is True).
        lr (float): Learning rate for the optimizer used in bounds optimization.
        lambda_valid (float): Coefficient for ensuring that the upper bound is greater than the lower bound.
        lambda_worst_case (float): Coefficient for weighting the worst-case loss.
    Methods:
        __init__(epsilon, optimize_bounds, gradient_iter=0, lr=0.1, lambda_valid=0.1, lambda_worst_case=10):
            Initializes the AffineNN object with the given parameters.
        get_bounds(x):
            Computes the affine arithmetic bounds for the input tensor `x` with perturbation `epsilon`.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                Interval: The computed interval bounds for the input tensor.
        tighten_up_intervals(z_L, z_U):
            Computes the loss for tightening interval bounds, ensuring valid and non-negative intervals.
            Args:
                z_L (torch.Tensor): Lower bounds of the interval.
                z_U (torch.Tensor): Upper bounds of the interval.
            Returns:
                torch.Tensor: The computed loss for interval tightening.
        _gradient_step(x, y):
            Performs a single gradient step to optimize the bounds using the input tensor 
            and target labels.
            Args:
                x (torch.Tensor): Input tensor.
                y (torch.Tensor): Target labels.
        forward(x, y):
            Computes the forward pass of the AffineNN method. If `optimize_bounds` is True, 
            performs gradient-based optimization to tighten bounds.
            Args:
                x (torch.Tensor): Input tensor.
                y (torch.Tensor): Target labels.
            Returns:
                Interval: The computed interval bounds after the forward pass.
    """

    def __init__(self, epsilon: float,
                 optimize_bounds: bool, 
                 gradient_iter: int = 0,
                 lr: float = 0.1,
                 lambda_valid: float = 0.1,
                 lambda_worst_case: float = 10.0
                 ):
        
        super().__init__()

        self.epsilon = epsilon
        self.optimize_bounds = optimize_bounds

        if optimize_bounds and gradient_iter == 0:
            raise ValueError("Gradient iteration must be greater than 0 when optimize_bounds is True.")
        
        self.gradient_iter = gradient_iter
        self.lambda_valid = lambda_valid
        self.lambda_worst_case = lambda_worst_case
        self.lr = lr

        log.info(f"Initialized Affine Arithmetic object with optimize_bounds={optimize_bounds}")

        
    def get_bounds(self, x: torch.Tensor) -> Interval:
        """
        Computes the bounds of an affine arithmetic expression through a neural network.
        This method propagates an affine expression through the layers of a neural network
        to compute the resulting interval bounds. It supports specific layer types such as
        `nn.Conv2d`, `nn.Linear`, and `nn.ReLU`.
        Args:
            x (float): The center value of the input interval.
        Returns:
            Tuple:
            Interval: The resulting interval bounds after propagating through the network.
            torch.Tensor: The accumulated error of the ReLU approximation.
        Raises:
            NotImplementedError: If the network contains a layer type that is not supported.
        """
        # WARNING: This method assumes that the input x is a single batch of data.
        # If x is a batch, we need to squeeze it to get the first element.
        x = x.squeeze(0)
        epsilon = self.epsilon * torch.ones_like(x)
        zl, zu = x - epsilon, x + epsilon
        expr = AffineExpr()
        interval = Interval(zl, zu)
        affine_func = expr.new_tensor(interval)

        relu_param_idx = 0  # Track which learnable ReLU param to use
        relu_loss = 0.0

        for m in self.module.module.children():
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    affine_func = affine_func.conv2d(layer)
                elif isinstance(layer, nn.Linear):
                    affine_func = affine_func.linear(layer)
                elif isinstance(layer, nn.ReLU):
                    if self.optimize_bounds:
                        slope = self.slope_relu_params[relu_param_idx]
                        affine_func, error = affine_func.relu(slope)
                        relu_loss += error
                        relu_param_idx += 1
                elif isinstance(layer, nn.Flatten):
                    affine_func = affine_func.flatten()
        return affine_func.to_interval(), relu_loss
    
    def tighten_up_intervals(self, z_L: torch.Tensor, z_U: torch.Tensor) -> torch.Tensor:
        """
        Tightens up the intervals by minimizing their width, ensuring validity.
        Args:
            z_L (torch.Tensor): The lower bounds of the intervals.
            z_U (torch.Tensor): The upper bounds of the intervals.
        Returns:
            torch.Tensor: The computed loss value, which is a combination of:
                - Interval width minimization loss.
                - Validity constraint loss ensuring lower bounds are less than or equal to upper bounds.
        """

        # Minimize interval width
        loss_tight = torch.mean(z_U - z_L)

        # Ensure lower bound <= upper bound
        loss_valid = self.lambda_valid * torch.mean(torch.clamp(z_L - z_U, min=0))
        
        return loss_tight + loss_valid
    
    def _gradient_step(self, x, y):
        """
        Performs a single gradient descent step for optimizing the model.
        Args:
            x (torch.Tensor): Input tensor for the model.
            y (torch.Tensor): Ground truth labels for the input data.
        Returns:
            None
        Description:
            This method computes the forward pass to obtain the bounds of the output
            intervals using `get_bounds`. It calculates the loss for tightening the
            intervals and the classification loss using the provided criterion. The
            total loss is computed as the sum of these two losses. The optimizer is
            then used to perform a gradient descent step to minimize the total loss.
        """
        outputs, relu_loss = self.get_bounds(x)
        loss = self.tighten_up_intervals(outputs.lower, outputs.upper)
        
        lb = outputs.lower.unsqueeze(0)
        ub = outputs.upper.unsqueeze(0)

        tmp = nn.functional.one_hot(y, lb.size(-1))
        z = torch.where(tmp.bool(), lb, ub)
        loss_cls = self.criterion(z, y)
        total_loss = self.lambda_worst_case * (loss_cls + relu_loss) + loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def forward(self, x, y):
        """
        Performs the forward pass for affine arithmetic calculations.
        Args:
            x (torch.Tensor): The input tensor for the affine arithmetic computation.
            y (torch.Tensor): The target or auxiliary tensor used in the computation.
        Returns:
            torch.Tensor: The computed bounds after applying affine arithmetic.
        Notes:
            - If `self.optimize_bounds` is True, the method performs gradient-based
              optimization for a specified number of iterations (`self.gradient_iter`)
              to refine the bounds.
            - The `get_bounds` method is used to compute the final bounds based on
              the input tensor and the epsilon tensor.
        """

        if self.optimize_bounds:
            self.slope_relu_params = nn.ParameterList()
            for m in self.module.module.children():
                for idx, layer in enumerate(m):
                    if isinstance(layer, nn.ReLU):
                        prev_layer = m[idx-1]
                        if isinstance(prev_layer, nn.Conv2d):
                            out_shape = self.module.layer_outputs.get(prev_layer)
                            slope = nn.Parameter(torch.log(0.5*torch.ones(out_shape)), requires_grad=True).to(DEVICE)
                        elif isinstance(prev_layer, nn.Linear):
                            slope = nn.Parameter(torch.log(0.5*torch.ones(prev_layer.out_features)), requires_grad=True).to(DEVICE)
                        self.slope_relu_params.append(slope)

            self.optimizer = torch.optim.Adam([*self.slope_relu_params], lr=self.lr)
            self.criterion = nn.CrossEntropyLoss()

            for _ in range(self.gradient_iter):            
                self._gradient_step(x, y)
        outputs, _ = self.get_bounds(x)
           
        return outputs

class AffineFunc:
    """
    AffineFunc is a class that represents affine arithmetic functions. It provides methods for performing arithmetic operations, 
    conversions, and transformations on affine functions, as well as utility methods for neural network operations like convolution 
    and linear transformations.
    Attributes:
        coeffs (torch.Tensor): The coefficients of the affine function.
        expr (AffineExpr): The affine expression associated with the function.
        device (str): "cpu" or "cuda". Default value set up to "cuda".
    Methods:
        __init__(shape: tuple = None, c: torch.Tensor = None, expr: 'AffineExpr' = None, device: str = "cpu"):
            Initializes an AffineFunc object with the given shape, coefficients, and expression.
        add_var(c, mask):
            Adds a variable to the affine expression.
        check_expr(other: 'AffineFunc') -> 'AffineExpr':
            Checks and returns a compatible affine expression between two AffineFunc objects.
        __add__(other):
            Adds another AffineFunc or a scalar to the current AffineFunc.
        __radd__(other):
            Implements reverse addition for scalar + AffineFunc.
        __sub__(other):
            Subtracts another AffineFunc, scalar, or tensor from the current AffineFunc.
        __rsub__(other):
            Implements reverse subtraction for scalar - AffineFunc.
        __mul__(other):
            Multiplies the current AffineFunc with another AffineFunc or a scalar.
        __rmul__(other):
            Implements reverse multiplication for scalar * AffineFunc.
        to_interval() -> Interval:
            Converts the affine function to an interval representation.
        relu(slope):
            Applies the ReLU activation function to the affine function with a given slope.
        softmax():
            Applies the softmax function to the affine function and computes its interval representation.
        conv2d(conv_layer: nn.Conv2d) -> 'AffineFunc':
            Applies a 2D convolution operation to the affine function using the given convolutional layer.
        linear(linear_layer: nn.Linear) -> 'AffineFunc':
            Applies a linear transformation to the affine function using the given linear layer.
    """

    def __init__(self, shape: tuple = None, c: torch.Tensor = None, expr: 'AffineExpr' = None):
        if c is None and expr is None:
            self.coeffs = torch.zeros(shape)
            self.expr = None
        elif c is not None and expr is not None:
            self.coeffs = c
            self.expr = expr
        elif c is not None:
            self.coeffs = c
            self.expr = None
        else:
            self.expr = expr
            self.coeffs = torch.zeros(shape)
        
        self.coeffs = self.coeffs.to(DEVICE)

    def add_var(self, c, mask):
        self.expr.add_var(self, c, mask)

    def check_expr(self, other: 'AffineFunc'):
        if self.expr == other.expr:
            return self.expr
        if self.expr is None:
            return other.expr
        if other.expr is None:
            return self.expr
        raise RuntimeError("AffineFunc::testAndSetExpr - incompatible expressions")
    
    def __add__(self, other):
        if isinstance(other, AffineFunc):
            result = AffineFunc(shape=self.coeffs.shape, expr=self.check_expr(other))
            result.coeffs = self.coeffs + other.coeffs
            return result
        else:
            result = AffineFunc(shape=self.coeffs.shape, expr=self.expr)
            result.coeffs = self.coeffs
            result.coeffs[..., 0] += other
            return result
        
    def __radd__(self, other):
        return self + other
        
    def __sub__(self, other):
        if isinstance(other, AffineFunc):
            result = AffineFunc(shape=self.coeffs.shape, expr=self.check_expr(other))
            result.coeffs = self.coeffs - other.coeffs
            return result
        elif isinstance(other, float):
            result = AffineFunc(shape=self.coeffs.shape, expr=self.expr)
            result.coeffs = self.coeffs
            result.coeffs[..., 0] -= other
            return result
        elif isinstance(other, torch.Tensor):
            result = AffineFunc(shape=self.coeffs.shape, expr=self.expr)
            result.coeffs = self.coeffs
            result.coeffs -= other
            return result
        
    def __rsub__(self, other):
        if isinstance(other, AffineFunc):
            result = AffineFunc(shape=self.coeffs.shape, expr=self.check_expr(other))
            result.coeffs = other.coeffs - self.coeffs
            return result
        else:
            result = AffineFunc(shape=self.coeffs.shape, expr=self.expr)
            result.coeffs = -self.coeffs
            result.coeffs[..., 0] += other
            return result

    def __mul__(self, other):
        if isinstance(other, AffineFunc):
            result = AffineFunc(self.check_expr(other))
            result.coeffs = self.coeffs * other.coeffs[..., 0]
            tmp = other.coeffs * self.coeffs[..., 0]
            tmp[..., 0] = 0.0
            result.coeffs += tmp
            ones = torch.ones(self.coeffs[:-1])
            I = Interval(-ones, ones)
            sf = I * torch.sum(self.coeffs[..., 1:], axis=-1)
            sg = I * torch.sum(other.coeffs[..., 1:], axis=-1)
            self.add_var(sf * sg, torch.ones_like(sf).astype(bool))
            return result
        else:
            result = AffineFunc(shape=self.coeffs.shape, expr=self.expr)
            result.coeffs = other * self.coeffs
            return result
        
    def __rmul__(self, other):
        return self * other
        
    def to_interval(self):
        ones = torch.ones((*self.coeffs.shape[:-1], self.coeffs.shape[-1] - 1)).to(device=DEVICE)
        I = Interval(-ones, ones)
        result = I * self.coeffs[..., 1:]
        result.lower = torch.sum(result.lower, axis=-1) + self.coeffs[..., 0]
        result.upper = torch.sum(result.upper, axis=-1) + self.coeffs[..., 0]
        
        return result

    def relu(self, slope):
        c = self.to_interval()

        mask1 = c <= 0
        mask2 = c >= 0
        mask3 = torch.logical_not(torch.logical_or(mask1, mask2))
    
        e = torch.exp(slope.unsqueeze(0)).squeeze(0)

        a0 = self.coeffs[..., 0]
        S = torch.sum(torch.abs(self.coeffs[..., 1:]), axis=-1)

        M = a0 + S
        B = 0.5 * e * M
        c = B/S
        D = torch.abs(B-c*a0)

        result = AffineFunc(shape=self.coeffs.shape, expr=self.expr)
        result.coeffs = c.unsqueeze(-1) * self.coeffs
        result.coeffs[..., 0] = B
        D = D[mask3]
        e = e[mask3]
        M = M[mask3]
        
        i1 = Interval(-D, -D)
        i2 = Interval((1-e)*M, (1-e)*M)
        hull_lower, hull_upper = interval_hull(i1, i2)
        
        error = (hull_upper - hull_lower).mean()

        result.coeffs[mask1] *= 0.0
        result.coeffs[mask2] = self.coeffs[mask2]
        result.add_var(Interval(hull_lower, hull_upper), mask3)
        return result, error
    
    def softmax(self):
        x = self.coeffs[..., 0]
        R = torch.max(torch.abs(x))
        s = F.softmax(x - R, dim=-1)

        g = AffineFunc(c=(self.coeffs.unsqueeze(1) - self.coeffs.unsqueeze(0)))
        g = (g.to_interval() - R + x).exp()
        g = g.sum(dim=1)
        g = (x - R).exp() / g

        dim = self.coeffs.shape[-2]
        Dg = [[None for _ in range(dim)] for _ in range(dim)]
        Dg_lower = torch.zeros((dim, dim))
        Dg_upper = torch.zeros((dim, dim))
        for i in range(dim):
            for c in range(i+1, dim):
                t = self.coeffs[..., i, :] + self.coeffs[..., c, :]
                t = t.unsqueeze(0).unsqueeze(0)
                tmp = AffineFunc(shape=self.coeffs.shape, expr=self.expr)
                tmp.coeffs = self.coeffs.unsqueeze(1) - self.coeffs.unsqueeze(0)
                tmp = tmp - t
                tmp = tmp.to_interval()
                tmp = (tmp - 2 * R + x[..., i] + x[..., c]).exp()
                tmp = tmp.sum(dim=(0, 1))
                t = (x[..., i] + x[..., c] - 2 * R).exp() / tmp
                lower = t.lower
                upper = t.upper
                Dg_lower[i, c] = Dg_lower[c, i] = lower 
                Dg_upper[i, c] = Dg_upper[c, i] = upper 

        for i in range(dim):
            t = (0.25-(g[i]-0.5)).sqr()
            lower = t.lower
            upper = t.upper
            Dg_lower[i, i] = lower
            Dg_upper[i, i] = upper

        Dg = Interval(Dg_lower, Dg_upper)
        
        x = x.unsqueeze(1)
        s = s.unsqueeze(1)
        r = s + Dg @ (self - x)

        # r.to_interval()
        ones = torch.ones((*self.coeffs.shape[:-1], self.coeffs.shape[-1] - 1))
        I = Interval(-ones, ones)
        result = I * r[..., 1:]
        result.lower = torch.sum(result.lower, axis=-1) + r.lower[..., 0]
        result.upper = torch.sum(result.upper, axis=-1) + r.upper[..., 0]
        g = intersection(result, g)
        assert torch.all(g.lower <= g.upper), "Empty intersection in Softmax"
        return g
    
    def conv2d(self, conv_layer: nn.Conv2d):
        x = self.coeffs.permute(3, 0, 1, 2)
        x = F.conv2d(
            x,
            weight=conv_layer.weight,
            bias=None,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups
        )
        x = x.permute(1, 2, 3, 0)

        if conv_layer.bias is not None:
            x[..., 0] += conv_layer.bias.view(-1, *([1] * (x.dim() - 2)))

        return AffineFunc(c=x, expr=self.expr)

    def linear(self, linear_layer: nn.Linear):
        A = linear_layer.weight
        b = linear_layer.bias
        
        x = self.coeffs
        x = torch.einsum("ea,oe->oa", x, A)
        
        if b is not None:
            x[..., 0] += b.view(*([1] * (x.dim() - 2)), -1)
            
        return AffineFunc(c=x, expr=self.expr)
    
    def flatten(self):
        x = self.coeffs
        x = x.flatten(start_dim=0, end_dim=-2)
        return AffineFunc(c=x, expr=self.expr)

class AffineExpr:
    """
    A class representing an affine expression, which is used in affine arithmetic
    to manage affine functions and their coefficients.
    Methods
    -------
    __init__():
        Initializes the AffineExpr object with a counter for tracking the current variable index.
    add_var(f: AffineFunc, c: Interval, mask: torch.Tensor = None):
        Adds a variable with an interval coefficient to the affine function. If a mask is provided,
        the operation is applied selectively based on the mask.
        Parameters:
        - f (AffineFunc): The affine function to which the variable is added.
        - c (Interval): The interval coefficient of the variable.
        - mask (torch.Tensor, optional): A tensor mask to selectively apply the operation.
    new_vector(u: Interval):
        Creates a new affine function vector initialized with the given interval.
        Parameters:
        - u (Interval): The interval used to initialize the new affine function vector.
        Returns:
        - AffineFunc: A new affine function vector.
    new_tensor(u: Interval):
        Creates a new affine function tensor initialized with the given interval.
        Parameters:
        - u (Interval): The interval used to initialize the new affine function tensor.
        Returns:
        - AffineFunc: A new affine function tensor.
    """

    def __init__(self):
        self.current = 0

    def add_var(self, f: AffineFunc, c: Interval, mask: torch.Tensor = None):
        """
        Adds a variable with an interval coefficient to the affine function.
        """
        if mask is None:
            c = c + f.coeffs[..., 0]
            mid, q = c.split()
            f.coeffs[..., 0] = mid
            r = torch.maximum(torch.abs(q.lower.flatten()), torch.abs(q.upper.flatten()))
            r = torch.diag(r).reshape((*c.lower.shape, r.numel()))
            self.current += c.lower.numel()
            f.coeffs = torch.concatenate((f.coeffs, r), axis=-1)
        else:
            mask = torch.nonzero(mask, as_tuple=True)
            mask_a0 = (*mask, torch.zeros_like(mask[0]))
            c = c + f.coeffs[mask_a0]
            mid, q = c.split()
            f.coeffs[mask_a0] = mid
            r = torch.maximum(torch.abs(q.lower), torch.abs(q.upper))
            r = torch.diag(r)
            new_var = torch.zeros((*f.coeffs.shape[:-1], c.lower.numel())).to(DEVICE)
            new_var[mask] = r
            self.current += c.lower.numel()
            f.coeffs = torch.concatenate((f.coeffs, new_var), axis=-1)

    def new_vector(self, u: Interval):
        shape = (*u.lower.shape, 1)
        result = AffineFunc(shape=shape, expr=self)
        self.add_var(result, u)
        return result
    
    def new_tensor(self, u: Interval):
        shape = (*u.lower.shape, 1)
        result = AffineFunc(shape=shape, expr=self)
        self.add_var(result, u)
        return result




###############################################

def test_vector_and_tensor_creation():
    # Test 1: Creating a vector
    expr = AffineExpr()
    interval = Interval(torch.tensor([1, 2]), torch.tensor([3, 4]))
    vector = expr.new_vector(interval)

    # Verify properties of the created vector
    assert vector.coeffs.shape == (2, 3)  # Shape should match the interval
    assert vector.expr == expr  # Expression should match the provided one
    print("Test 1 passed: Vector creation works correctly.")

    expr = AffineExpr()
    interval = Interval(
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[5, 6], [7, 8]])
    )
    tensor = expr.new_tensor(interval)

    # Verify properties of the created tensor
    assert tensor.coeffs.shape == (2, 2, 5)  # Shape should match the interval
    assert tensor.expr == expr  # Expression should match the provided one
    print("Test 2 passed: Tensor creation works correctly.")

    tensor.to_interval()
    tensor.relu()
    vector.softmax()

# Run the tests
# test_vector_and_tensor_creation()