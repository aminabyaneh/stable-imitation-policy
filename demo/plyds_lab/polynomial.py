"""Explicit monomial maps; every polynomial coefficient is matched."""
import numpy as np
from scipy import sparse


def basis(degree, constant=False):
    return [(a, total-a) for total in range(0 if constant else 1, degree+1)
            for a in range(total+1)]


def features(x, powers):
    x = np.atleast_2d(x)
    powers = np.asarray(powers, dtype=int)
    with np.errstate(over="ignore", invalid="ignore"):
        return x[:, :1]**powers[:, 0] * x[:, 1:2]**powers[:, 1]


def gradient_features(x, powers, axis):
    powers = np.asarray(powers, dtype=int)
    derivative = powers.copy()
    factor = derivative[:, axis].copy()
    derivative[:, axis] = np.maximum(0, derivative[:, axis]-1)
    return features(x, derivative)*factor


def gram_map(powers, coefficient_basis):
    lookup = {m: i for i, m in enumerate(coefficient_basis)}
    rows, cols = [], []
    size = len(powers)
    for i, (a, b) in enumerate(powers):
        for j, (c, d) in enumerate(powers):
            rows.append(lookup[(a+c, b+d)]); cols.append(i*size+j)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(coefficient_basis), size*size))


def derivative_policy_map(vcoeff, vpowers, fpowers, cpowers):
    """Maps component-major policy coefficients to grad(V).f for fixed V."""
    lookup = {m:i for i,m in enumerate(cpowers)}
    matrix = np.zeros((len(cpowers), 2*len(fpowers)))
    for vc, vexp in zip(vcoeff, vpowers):
        for axis in range(2):
            if not vexp[axis]: continue
            for j, fexp in enumerate(fpowers):
                exp = [vexp[k]+fexp[k]-(k==axis) for k in range(2)]
                matrix[lookup[tuple(exp)], axis*len(fpowers)+j] += vc*vexp[axis]
    return sparse.csr_matrix(matrix)


def derivative_v_map(coeff, fpowers, vpowers, cpowers):
    """Maps V coefficients to grad(V).f for a fixed policy."""
    lookup = {m:i for i,m in enumerate(cpowers)}
    matrix = np.zeros((len(cpowers),len(vpowers)))
    for j,vexp in enumerate(vpowers):
        for axis in range(2):
            if not vexp[axis]: continue
            for fcoef,fexp in zip(coeff[:,axis],fpowers):
                exp = tuple(vexp[k]+fexp[k]-(k==axis) for k in range(2))
                matrix[lookup[exp],j] += vexp[axis]*fcoef
    return sparse.csr_matrix(matrix)
