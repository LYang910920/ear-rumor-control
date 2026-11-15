"""Coupled sharing and emotion dynamics. Matrix entry [g,h] means h influences g."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

Array = NDArray[np.float64]
Forcing = Callable[[float], tuple[Array, Array]]

@dataclass
class Model:
    A: object
    B: object
    pi: Array
    beta: Array
    alpha: Array
    delta: Array
    eta: Array
    gamma: Array
    mu: Array
    u_max: Array
    v_max: Array
    resource_u: Array
    resource_v: Array
    budget: float
    weight_x: Array
    weight_y: Array
    cost_u: Array
    cost_v: Array
    terminal_x: Array
    terminal_y: Array
    source_x_max: Array
    source_y_max: Array

    def __post_init__(self) -> None:
        self.pi = np.asarray(self.pi, dtype=float)
        if self.pi.ndim != 1 or not np.all(np.isfinite(self.pi)):
            raise ValueError('pi must be a finite one-dimensional probability vector.')
        self.K = len(self.pi)
        if self.K == 0 or np.any(self.pi <= 0) or not np.isclose(self.pi.sum(), 1):
            raise ValueError('pi must be a strictly positive probability vector.')
        for name in ('A', 'B'):
            matrix = getattr(self, name)
            matrix = sparse.csr_matrix(matrix, dtype=float) if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
            values = matrix.data if sparse.issparse(matrix) else matrix
            if matrix.shape != (self.K, self.K) or np.any(values < 0) or not np.all(np.isfinite(values)):
                raise ValueError(f'{name} must be a finite nonnegative K by K matrix.')
            setattr(self, name, matrix)
        for name in self.__dataclass_fields__:
            if name in ('A', 'B', 'pi', 'budget'):
                continue
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim == 0:
                value = np.full(self.K, float(value))
            if value.shape != (self.K,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f'{name} must be nonnegative, finite, and scalar or length K.')
            setattr(self, name, value)
        for name in ('beta', 'delta', 'mu', 'cost_u', 'cost_v', 'resource_u', 'resource_v'):
            if np.any(getattr(self, name) <= 0):
                raise ValueError(f'{name} must be strictly positive.')
        if np.any(self.u_max > 1) or self.budget < 0 or not np.isfinite(self.budget):
            raise ValueError('Require u_max <= 1 and finite budget >= 0.')

    def zero_forcing(self, t: float) -> tuple[Array, Array]:
        return np.zeros(self.K), np.zeros(self.K)

    def check_forcing(self, sx: Array, sy: Array) -> None:
        for name, z, upper in [('rumor', sx, self.source_x_max), ('emotion', sy, self.source_y_max)]:
            z = np.asarray(z)
            if z.shape != (self.K,) or np.any(~np.isfinite(z)) or np.any(z < 0) or np.any(z > upper + 1e-12):
                raise ValueError(f'{name} forcing violates the declared bounds.')

    def quantities(self, state: Array, control: Array, sx: Array, sy: Array):
        x, y = np.split(state, 2)
        u, v = np.split(control, 2)
        pressure = self.A @ x + sx
        delivered_input = (1-u)*pressure
        rate = self.beta*(1+self.alpha*y)
        emotional_input = self.eta*(self.B @ y)+self.gamma*delivered_input+sy
        return x, y, u, v, pressure, delivered_input, rate, emotional_input

    def rhs(self, state: Array, control: Array, sx: Array, sy: Array) -> Array:
        x,y,u,v,p,e,c,b = self.quantities(state, control, sx, sy)
        return np.concatenate(((1-x)*c*e-self.delta*x,
                               (1-y)*b-(self.mu+v)*y))

    def rhs_vjp(self, state: Array, control: Array, sx: Array, sy: Array, adjoint: Array) -> tuple[Array, Array]:
        """Exact products (df/dz)^T adjoint and (df/da)^T adjoint."""
        x,y,u,v,p,e,c,b = self.quantities(state, control, sx, sy)
        lx,ly = np.split(adjoint, 2)
        common = (1-x)*c*lx+(1-y)*self.gamma*ly
        gx = self.A.T @ ((1-u)*common) - (c*e+self.delta)*lx
        gy = ((1-x)*self.beta*self.alpha*e*lx
              +self.B.T @ ((1-y)*self.eta*ly) -(b+self.mu+v)*ly)
        gu = -p*common
        gv = -y*ly
        return np.concatenate((gx,gy)), np.concatenate((gu,gv))

    def running_cost(self, state: Array, control: Array) -> float:
        x,y = np.split(state,2); u,v = np.split(control,2)
        return float(self.pi @ (self.weight_x*x+self.weight_y*y
                     +.5*self.cost_u*u*u+.5*self.cost_v*v*v))

    def running_gradient(self, state: Array, control: Array) -> tuple[Array,Array]:
        u,v = np.split(control,2)
        return (np.concatenate((self.pi*self.weight_x,self.pi*self.weight_y)),
                np.concatenate((self.pi*self.cost_u*u,self.pi*self.cost_v*v)))

    def terminal_cost(self, state: Array) -> float:
        return float(self.terminal_gradient() @ state)

    def terminal_gradient(self) -> Array:
        return np.concatenate((self.pi*self.terminal_x,self.pi*self.terminal_y))

    def upper_controls(self) -> Array:
        return np.concatenate((self.u_max,self.v_max))

    def resource_vector(self) -> Array:
        return np.concatenate((self.pi*self.resource_u,self.pi*self.resource_v))

    def step_limit(self) -> float:
        """SSPRK3 preserves [0,1]^(2K) for h <= this uniform Euler bound."""
        ar = np.asarray(self.A.sum(axis=1)).ravel()
        br = np.asarray(self.B.sum(axis=1)).ravel()
        rx = self.beta*(1+self.alpha)*(ar+self.source_x_max)+self.delta
        ry = self.eta*br+self.gamma*(ar+self.source_x_max)+self.source_y_max+self.mu+self.v_max
        return 1/float(np.maximum(rx,ry).max())


def make_model(A, B, pi=None, **overrides) -> Model:
    K = A.shape[0]
    default = dict(pi=np.ones(K)/K if pi is None else pi,
        beta=.12, alpha=1.5, delta=.22, eta=.03, gamma=.06, mu=.06,
        u_max=.85, v_max=.20, resource_u=1., resource_v=3., budget=.65,
        weight_x=1., weight_y=.25, cost_u=.08, cost_v=1.5,
        terminal_x=2., terminal_y=1., source_x_max=0., source_y_max=0.)
    default.update(overrides)
    return Model(A=A,B=B,**default)
