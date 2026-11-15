"""Emotional upper bounds and the longest eligible first hold (Algorithm 3)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, linprog
from scipy import sparse
from .model import Model, Array


def dense(a): return a.toarray() if sparse.issparse(a) else np.asarray(a)


def emotional_bound(model: Model, y0: Array, action: Array, duration: float) -> tuple[Array,Array]:
    if not np.isfinite(duration) or duration<0: raise ValueError('duration must be finite and nonnegative.')
    y0=np.asarray(y0,dtype=float); action=np.asarray(action,dtype=float)
    if y0.shape!=(model.K,) or np.any(~np.isfinite(y0)) or np.any((y0<0)|(y0>1)):
        raise ValueError('Emotional state must be finite and lie in the unit box.')
    if action.shape!=(2*model.K,) or np.any(~np.isfinite(action)) or np.any(action<0) or np.any(action>model.upper_controls()+1e-12):
        raise ValueError('Action must respect its declared box.')
    u,v=np.split(action,2)
    ar=np.asarray(model.A.sum(axis=1)).ravel(); br=np.asarray(model.B.sum(axis=1)).ravel()
    H=model.eta*br+model.gamma*(1-u)*(ar+model.source_x_max)+model.source_y_max
    d=model.mu+v; decay=H+d; equilibrium=H/decay
    endpoint=equilibrium+(y0-equilibrium)*np.exp(-decay*duration)
    return endpoint,np.maximum(y0,endpoint)


def maximum_hold(model: Model, y0: Array, ceiling: Array, action: Array) -> float:
    """Largest duration for which every emotional upper bound stays below its ceiling."""
    y0=np.asarray(y0,dtype=float); ceiling=np.asarray(ceiling,dtype=float)
    emotional_bound(model,y0,action,0.)
    if ceiling.shape!=(model.K,) or np.any(~np.isfinite(ceiling)) or np.any(ceiling<0):
        raise ValueError('Ceiling must be a finite nonnegative vector.')
    if np.any(ceiling<y0) or np.any(ceiling>1): return 0.
    u,v=np.split(action,2)
    ar=np.asarray(model.A.sum(axis=1)).ravel(); br=np.asarray(model.B.sum(axis=1)).ravel()
    H=model.eta*br+model.gamma*(1-u)*(ar+model.source_x_max)+model.source_y_max
    k=H+model.mu+v; equilibrium=H/k
    answer=np.full(model.K,np.inf)
    rising=equilibrium>ceiling
    answer[rising]=np.log((equilibrium[rising]-y0[rising])/(equilibrium[rising]-ceiling[rising]))/k[rising]
    return max(0.,float(answer.min()))


def rumor_matrix(model: Model, ceiling: Array, action: Array) -> Array:
    u,_=np.split(action,2)
    return (model.beta*(1-u)*(1+model.alpha*ceiling))[:,None]*dense(model.A)-np.diag(model.delta)


def contraction_margin(model: Model, ceiling: Array, action: Array, q: Array, rate: float) -> Array:
    return q @ rumor_matrix(model,ceiling,action)+rate*q


def source_bound(model: Model, ceiling: Array, action: Array, q: Array) -> float:
    u,_=np.split(action,2)
    return float(q @ (model.beta*(1-u)*(1+model.alpha*ceiling)*model.source_x_max))


def positive_weight(model: Model, auxiliary_action: Array, ceiling: Array|None=None) -> Array:
    M=rumor_matrix(model,np.ones(model.K) if ceiling is None else ceiling,auxiliary_action)
    if np.max(np.linalg.eigvals(M).real)>=-1e-10:
        raise ValueError('Auxiliary rumor matrix must have negative spectral abscissa.')
    q=np.linalg.solve(-M.T,np.ones(model.K))
    if np.any(q<=0): raise ArithmeticError('Positive weight construction failed.')
    return q/q.sum()


def worst_case_linear_constraints(model: Model, q: Array, rate: float):
    # K inequalities: sum_g q_g beta_g(1+alpha_g) A_gh (1-u_g)
    #                  <= (delta_h-rate) q_h.
    C=(q*model.beta*(1+model.alpha))[:,None]*dense(model.A)
    rhs=(model.delta-rate)*q-C.sum(axis=0)
    L=np.zeros((model.K,2*model.K)); L[:,:model.K]=-C.T
    return L,rhs


@dataclass
class HoldDecision:
    action: Array|None
    duration: float|None
    mode: str
    ceiling: Array|None
    residual_upper_bound: float|None


def select_interval(model: Model, y0: Array, proposed: Array, durations: Array,
                    q: Array, rate: float, allow_adjustment: bool=True) -> HoldDecision:
    """Choose the longest eligible hold; adjust the proposed action when needed."""
    proposed=np.asarray(proposed,dtype=float); q=np.asarray(q,dtype=float)
    durations=np.asarray(durations,dtype=float)
    if (q.shape!=(model.K,) or np.any(~np.isfinite(q)) or np.any(q<=0)
        or not np.isfinite(rate) or rate<=0 or durations.ndim!=1 or durations.size==0
        or np.any(~np.isfinite(durations)) or np.any(durations<=0)):
        raise ValueError('Require finite positive rate, weight vector, and nonempty duration menu.')
    emotional_bound(model,y0,proposed,0.)
    if model.resource_vector()@proposed>model.budget+1e-12:
        raise ValueError('The proposed action must be admissible.')
    choices=sorted(set(float(x) for x in durations),reverse=True)
    for duration in choices:
        _,Y=emotional_bound(model,y0,proposed,duration)
        if (np.all(contraction_margin(model,Y,proposed,q,rate)<=0.)
            and model.resource_vector()@proposed<=model.budget):
            return HoldDecision(proposed.copy(),duration,'nominal',Y,
                                source_bound(model,Y,proposed,q)/rate)
    if not allow_adjustment:
        return HoldDecision(None,None,'no_eligible_nominal_hold',None,None)
    L,rhs=worst_case_linear_constraints(model,q,rate)
    C=np.vstack((L,model.resource_vector())); b=np.r_[rhs,model.budget]
    bounds=list(zip(np.zeros(2*model.K),model.upper_controls()))
    lp_options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    feasible=linprog(np.zeros(2*model.K),A_ub=C,b_ub=b,bounds=bounds,
                    method='highs',options=lp_options)
    if not feasible.success:
        mode='infeasible' if feasible.status==2 else 'numerical_failure'
        return HoldDecision(None,None,mode,None,None)
    Y=np.ones(model.K)
    def original_ok(a):
        return (np.all(a>=0) and np.all(a<=model.upper_controls())
                and np.all(contraction_margin(model,Y,a,q,rate)<=0.)
                and model.resource_vector()@a<=model.budget)
    def solve_qp(start,limits):
        return minimize(lambda a:.5*np.sum((a-proposed)**2),start,
                        jac=lambda a:a-proposed,method='SLSQP',bounds=bounds,
                        constraints={'type':'ineq','fun':lambda a:limits-C@a,
                                     'jac':lambda a:-C},
                        options={'ftol':1e-12,'maxiter':300})
    sol=solve_qp(feasible.x,b)
    if sol.success:
        action=np.clip(sol.x,0,model.upper_controls())
        if original_ok(action):
            return HoldDecision(action,choices[0],'adjusted',Y,
                                source_bound(model,Y,action,q)/rate)
    # Move active constraints inward by the fixed rounding margin.
    inward=1e-9*np.maximum(1.,np.abs(b)); tightened=b-inward
    strict=linprog(np.zeros(2*model.K),A_ub=C,b_ub=tightened,bounds=bounds,
                   method='highs',options=lp_options)
    if strict.success:
        sol=solve_qp(strict.x,tightened)
        if sol.success:
            action=np.clip(sol.x,0,model.upper_controls())
            if original_ok(action):
                return HoldDecision(action,choices[0],'adjusted',Y,
                                    source_bound(model,Y,action,q)/rate)
    return HoldDecision(None,None,'numerical_failure',None,None)
