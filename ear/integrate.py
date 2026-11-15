"""Positive third-order state integration and the matching discrete adjoint."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .model import Model, Array, Forcing

@dataclass
class Step:
    phase: int
    h: float
    states: tuple[Array,Array,Array]
    sources: tuple[tuple[Array,Array],tuple[Array,Array],tuple[Array,Array]]

@dataclass
class Trajectory:
    times: Array
    states: Array
    cost: float
    tape: list[Step]


def integrate(model: Model, initial: Array, controls: Array, grid: Array,
              max_step: float=.05, forcing: Forcing|None=None,
              keep_tape: bool=False, *, store_history: bool=True,
              observer: Callable[[float,Array],None]|None=None) -> Trajectory:
    if keep_tape and not store_history:
        raise ValueError('Adjoint integration requires stored stages and history.')
    grid=np.asarray(grid,dtype=float); controls=np.asarray(controls,dtype=float)
    z=np.asarray(initial,dtype=float).copy()
    if grid.ndim!=1 or len(grid)<2 or np.any(~np.isfinite(grid)) or np.any(~np.isfinite(controls)) or np.any(np.diff(grid)<=0) or controls.shape!=(len(grid)-1,2*model.K):
        raise ValueError('grid and phase-control dimensions do not agree.')
    if z.shape!=(2*model.K,) or np.any(~np.isfinite(z)) or np.any(z<0) or np.any(z>1):
        raise ValueError('The initial state must lie in the unit box.')
    if np.any(controls < -1e-12) or np.any(controls > model.upper_controls()+1e-12):
        raise ValueError('Control bounds violated.')
    if np.any(controls @ model.resource_vector() > model.budget+1e-10):
        raise ValueError('Per-phase resource constraint violated.')
    if not np.isfinite(max_step) or max_step<=0:
        raise ValueError('max_step must be finite and positive.')
    # This step is fixed before optimization, uniformly for all admissible actions.
    step=min(max_step, .9*model.step_limit())
    source=forcing or model.zero_forcing
    zero_sources=((np.zeros(model.K),np.zeros(model.K)),)*3 if forcing is None else None
    times=[float(grid[0])]; history=[z.copy()]; tape=[]; cost=0.
    if observer is not None: observer(float(grid[0]),z)
    for m,(left,right) in enumerate(zip(grid[:-1],grid[1:])):
        internal=[float(x) for x in getattr(source,'breakpoints',()) if left<float(x)<right]
        cuts=sorted(set([float(left),*internal,float(right)]))
        for seg_left,seg_right in zip(cuts[:-1],cuts[1:]):
            count=int(np.ceil((seg_right-seg_left)/step)); h=(seg_right-seg_left)/count
            a=controls[m]
            for j in range(count):
                t=seg_left+j*h
                right_step=seg_right if j==count-1 else seg_left+(j+1)*h
                # Evaluate the right stage on the left of a known source jump.
                stage_times=(t,np.nextafter(right_step,t),t+.5*h)
                ss=zero_sources
                if ss is None:
                    ss=tuple(tuple(np.asarray(q,dtype=float) for q in source(float(ts))) for ts in stage_times)
                    for sx,sy in ss: model.check_forcing(sx,sy)
                z1=z.copy(); k1=model.rhs(z1,a,*ss[0])
                z2=z+h*k1; k2=model.rhs(z2,a,*ss[1])
                z3=z+.25*h*(k1+k2); k3=model.rhs(z3,a,*ss[2])
                nxt=z+h*(k1/6+k2/6+2*k3/3)
                if any(np.any(s < -1e-10) or np.any(s > 1+1e-10) for s in (z2,z3,nxt)):
                    raise FloatingPointError('State left the unit box.')
                cost+=h*(model.running_cost(z1,a)/6+model.running_cost(z2,a)/6+2*model.running_cost(z3,a)/3)
                if keep_tape: tape.append(Step(m,h,(z1,z2,z3),ss))
                z=nxt
                if store_history:
                    times.append(float(right_step)); history.append(z.copy())
                if observer is not None: observer(float(right_step),z)
    if not store_history:
        times.append(float(grid[-1])); history.append(z.copy())
    cost+=model.terminal_cost(z)
    return Trajectory(np.asarray(times),np.asarray(history),float(cost),tape)


def objective_gradient(model: Model, initial: Array, controls: Array, grid: Array,
                       max_step: float=.05, forcing: Forcing|None=None) -> tuple[float,Array,Trajectory]:
    controls=np.asarray(controls,dtype=float); grid=np.asarray(grid,dtype=float)
    tr=integrate(model,initial,controls,grid,max_step,forcing,keep_tape=True)
    gradient=np.zeros_like(controls); p=model.terminal_gradient()
    for item in reversed(tr.tape):
        m,h=item.phase,item.h; a=controls[m]
        s1,s2,s3=item.states
        weights=(h/6,h/6,2*h/3)
        kbars=[weights[j]*p for j in range(3)]
        base=p.copy(); gu=np.zeros(2*model.K)
        # s3 = z + h*(k1+k2)/4
        gz,ga=model.rhs_vjp(s3,a,*item.sources[2],kbars[2])
        lz,la=model.running_gradient(s3,a)
        bar3=gz+weights[2]*lz; gu+=ga+weights[2]*la
        base+=bar3; kbars[0]+=h*.25*bar3; kbars[1]+=h*.25*bar3
        # s2 = z + h*k1
        gz,ga=model.rhs_vjp(s2,a,*item.sources[1],kbars[1])
        lz,la=model.running_gradient(s2,a)
        bar2=gz+weights[1]*lz; gu+=ga+weights[1]*la
        base+=bar2; kbars[0]+=h*bar2
        # s1 = z
        gz,ga=model.rhs_vjp(s1,a,*item.sources[0],kbars[0])
        lz,la=model.running_gradient(s1,a)
        base+=gz+weights[0]*lz; gu+=ga+weights[0]*la
        gradient[m]+=gu; p=base
    return tr.cost,gradient,tr
