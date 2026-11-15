"""State and cumulative-activity error bounds for group approximation (Appendix F)."""
import numpy as np
from scipy.special import exprel


def _check(time,L,defect,initial_error):
    t=np.asarray(time,dtype=float)
    if (np.any(~np.isfinite(t)) or np.any(t<0)
        or not all(np.isfinite(v) and v>=0 for v in (L,defect,initial_error))):
        raise ValueError('Bound arguments must be finite and nonnegative.')
    return t


def state_error(time,L,defect,initial_error=0.):
    t=_check(time,L,defect,initial_error)
    result=np.zeros_like(t)
    if initial_error==0 and defect==0: return result
    with np.errstate(over='ignore',invalid='ignore'):
        if initial_error>0: result=result+initial_error*np.exp(L*t)
        if defect>0: result=result+defect*t*exprel(L*t)
    return result


def integrated_error(T,L,defect,initial_error=0.):
    T=_check(T,L,defect,initial_error)
    if T.ndim!=0: raise ValueError('T must be scalar.')
    T=float(T)
    if T==0 or (initial_error==0 and defect==0): return 0.
    z=L*T
    with np.errstate(over='ignore'):
        phi2=(.5+z/6+z*z/24+z**3/120+z**4/720
              if z<1e-3 else (np.expm1(z)-z)/(z*z))
        value=0.
        if initial_error>0: value+=initial_error*T*exprel(z)
        if defect>0: value+=defect*T*T*phi2
    return float(value)


def quotient_constants(node_model,epsilon_A,epsilon_B):
    """Compute Lipschitz and contact-defect constants from the individual-node model."""
    if any(not np.isfinite(e) or e<0 for e in (epsilon_A,epsilon_B)):
        raise ValueError('Contact residuals must be finite and nonnegative.')
    m=node_model
    ar=np.asarray(m.A.sum(axis=1)).ravel(); br=np.asarray(m.B.sum(axis=1)).ravel()
    lx=m.beta*(1+m.alpha)*(2*ar+m.source_x_max)+m.delta+m.beta*m.alpha*(ar+m.source_x_max)
    ly=2*m.eta*br+2*m.gamma*ar+m.gamma*m.source_x_max+m.source_y_max+m.mu+m.v_max
    L=float(np.maximum(lx,ly).max())
    defect=max(float((m.beta*(1+m.alpha)).max()*epsilon_A),
               float(m.eta.max()*epsilon_B+m.gamma.max()*epsilon_A))
    return L,defect
