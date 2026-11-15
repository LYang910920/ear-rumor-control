"""Budget projection and Armijo projected gradient for held interventions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .model import Model, Array, Forcing
from .integrate import integrate, objective_gradient, Trajectory


def project_action(z: Array, upper: Array, resources: Array, budget: float) -> Array:
    z=np.asarray(z,dtype=float); upper=np.asarray(upper,dtype=float); a=np.asarray(resources,dtype=float)
    if (z.ndim!=1 or z.size==0 or z.shape!=upper.shape or z.shape!=a.shape
        or not all(np.all(np.isfinite(x)) for x in (z,upper,a))
        or np.any(upper<0) or np.any(a<=0) or not np.isfinite(budget) or budget<0):
        raise ValueError('Invalid projection arguments.')
    y=np.clip(z,0,upper)
    if a@y<=budget: return y
    if budget==0: return np.zeros_like(z)
    lo=0.; hi=max(0.,float(np.max(z/a)))
    for _ in range(80):
        mid=(lo+hi)/2; y=np.clip(z-mid*a,0,upper)
        if a@y>budget: lo=mid
        else: hi=mid
    return np.clip(z-hi*a,0,upper)


def project_schedule(U: Array, model: Model) -> Array:
    U=np.asarray(U,dtype=float)
    upper=model.upper_controls(); a=model.resource_vector()
    if U.ndim!=2 or U.shape[1]!=len(upper) or len(U)==0 or not np.all(np.isfinite(U)):
        raise ValueError('Schedule must be a nonempty finite matrix with 2K columns.')
    Y=np.clip(U,0,upper)
    if model.budget==0: return np.zeros_like(U)
    active=Y@a>model.budget
    if not np.any(active): return Y
    Z=U[active];lo=np.zeros(len(Z));hi=np.maximum(0,np.max(Z/a,axis=1))
    # Independent scalar multipliers, processed together; same Euclidean projection.
    for _ in range(80):
        mid=(lo+hi)/2; v=np.clip(Z-mid[:,None]*a,0,upper)
        over=v@a>model.budget
        lo=np.where(over,mid,lo);hi=np.where(over,hi,mid)
    Y[active]=np.clip(Z-hi[:,None]*a,0,upper)
    return Y

@dataclass
class Solution:
    controls: Array
    trajectory: Trajectory
    iterations: int
    projected_step: float
    status: str


def optimize(model: Model, initial: Array, grid: Array, initial_controls: Array|None=None,
             max_step: float=.05, max_iterations: int=2000, tolerance: float=1e-3,
             theta: float|None=None, forcing: Forcing|None=None) -> Solution:
    """Minimize the whole-interval objective with projected Armijo steps (Algorithm 2)."""
    grid=np.asarray(grid,dtype=float)
    if grid.ndim!=1 or len(grid)<2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid)<=0):
        raise ValueError('Review grid must be finite and strictly increasing.')
    if not isinstance(max_iterations,(int,np.integer)) or max_iterations<1:
        raise ValueError('max_iterations must be a positive integer.')
    M=len(grid)-1
    U=np.zeros((M,2*model.K)) if initial_controls is None else project_schedule(initial_controls,model)
    if U.shape!=(M,2*model.K): raise ValueError('Initial schedule has wrong shape.')
    theta=float(theta if theta is not None else model.K/max(float(np.mean(np.diff(grid))),1e-6))
    if not np.isfinite(theta) or not np.isfinite(tolerance) or theta<=0 or tolerance<=0:
        raise ValueError('theta and tolerance must be positive.')
    status='iteration_limit'
    scale=np.maximum(model.upper_controls(),1e-12)
    for it in range(max_iterations+1):
        J,G,tr=objective_gradient(model,initial,U,grid,max_step,forcing)
        d=project_schedule(U-theta*G,model)-U
        residual=float(np.max(np.abs(d)/scale))
        if residual<=tolerance:
            status='stationary_tolerance';break
        if it==max_iterations: break
        slope=float(np.sum(G*d))
        if slope>=0: raise ArithmeticError('Projection did not give a descent direction.')
        alpha=1.;accepted=False
        for _ in range(40):
            trial=U+alpha*d
            trial_tr=integrate(model,initial,trial,grid,max_step,forcing)
            if trial_tr.cost<=J+1e-4*alpha*slope:
                U=trial;accepted=True;break
            alpha*=.5
        if not accepted:
            status='line_search_limit'
            break
    return Solution(U,tr,it,residual,status)
