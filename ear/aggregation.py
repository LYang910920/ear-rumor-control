"""Joint-degree grouping and population-weighted contact matrices (Algorithm 1)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class Quotient:
    A: np.ndarray
    B: np.ndarray
    pi: np.ndarray
    membership: np.ndarray
    epsilon_A: float | None = None
    epsilon_B: float | None = None


def joint_degree_classes(features: np.ndarray, max_groups: int=128) -> np.ndarray:
    """Split log-degree features at observed medians, keeping equal values together."""
    f=np.log1p(np.asarray(features,dtype=float))
    if f.ndim!=2 or len(f)==0 or np.any(~np.isfinite(f)) or np.any(f<0) or max_groups<1:
        raise ValueError('Need a nonempty nonnegative feature matrix and max_groups >= 1.')
    groups=[np.arange(len(f),dtype=np.int64)]
    frozen=set()
    while len(groups)<max_groups:
        candidates=[i for i in range(len(groups)) if i not in frozen and len(groups[i])>1]
        if not candidates: break
        i=max(candidates,key=lambda j:(len(groups[j]),-j)); ids=groups[i]
        spread=np.ptp(f[ids],axis=0); axis=int(np.argmax(spread))
        if spread[axis]==0: frozen.add(i); continue
        vals=f[ids,axis]; values=np.unique(vals)
        # Cut at a distinct observed value nearest the sample median, leaving both sides nonempty.
        med=np.median(vals); eligible=values[:-1]
        cut=eligible[int(np.argmin(np.abs(eligible-med)))]
        left=ids[vals<=cut]; right=ids[vals>cut]
        if len(left)==0 or len(right)==0: raise ArithmeticError('Empty partition cell.')
        groups[i]=left; groups.append(right)
    membership=np.empty(len(f),dtype=np.int32)
    for k,ids in enumerate(groups): membership[ids]=k
    return membership


def capped_contacts(adjacency, reference_degree: float=20., max_contacts: float=5.):
    if reference_degree<=0 or max_contacts<=0: raise ValueError('Contact scales must be positive.')
    a=sparse.csr_matrix(adjacency,dtype=float)
    if np.any(a.data<0): raise ValueError('Negative contacts.')
    degree=np.asarray(a.sum(axis=1)).ravel()
    activity=np.minimum(degree/reference_degree,max_contacts)
    weights=np.divide(activity,degree,out=np.zeros_like(activity),where=degree>0)
    return sparse.diags(weights)@a


def quotient_from_sparse(A,B,membership=None,max_groups=128,compute_residual=True)->Quotient:
    A=sparse.csr_matrix(A); B=sparse.csr_matrix(B)
    if A.shape!=B.shape or A.shape[0]!=A.shape[1]: raise ValueError('Layer shapes differ.')
    N=A.shape[0]
    if membership is None:
        feat=np.c_[A.getnnz(axis=1),A.getnnz(axis=0),B.getnnz(axis=1),B.getnnz(axis=0)]
        membership=joint_degree_classes(feat,max_groups)
    membership=np.asarray(membership,dtype=np.int32)
    if membership.shape!=(N,) or np.any(membership<0): raise ValueError('Invalid membership.')
    _,membership=np.unique(membership,return_inverse=True)
    K=int(membership.max()+1); counts=np.bincount(membership,minlength=K)
    P=sparse.csr_matrix((np.ones(N),(np.arange(N),membership)),shape=(N,K))
    H=sparse.diags(1/counts)@P.T
    AP=A@P; BP=B@P
    Aq=(H@AP).toarray(); Bq=(H@BP).toarray()
    eps=[]
    for mat,q in ((AP,Aq),(BP,Bq)):
        if not compute_residual: eps.append(None); continue
        # Block evaluation avoids materializing N by K dense arrays.
        value=0.
        for left in range(0,N,4096):
            right=min(N,left+4096)
            value=max(value,float(np.abs(mat[left:right].toarray()-q[membership[left:right]]).sum(axis=1).max()))
        eps.append(value)
    return Quotient(Aq,Bq,counts/N,membership,eps[0],eps[1])
