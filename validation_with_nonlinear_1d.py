import numpy as np
import scipy.linalg as sl
import sympy as sym
import cheby_tools.cheby1d as c1d
from cheby_tools.cheby1d import add, mul, scale, sub

def f_orig(x):
    return -np.sin(x)

def g_orig(x):
    return 1.0

def l_orig(x):
    return x ** 2.0

if __name__ == "__main__":
    N = 9
    indices = c1d.generate_indices(N)
    LEN_INDICES = len(indices)

    f = c1d.expand(f_orig, indices)
    g = c1d.expand(g_orig, indices)
    l = c1d.expand(l_orig, indices)
    u = c1d.expand(lambda x: 0.0, indices)

    v = c1d.of_dict({idx: sym.Symbol("v_{}".format(idx), real=True) for idx in indices})
    dv = c1d.diff(v)

    LOOP = 20
    for _ in range(LOOP):
        u_sq = mul(u, u)
        ghjb_lhs = mul(dv, add(f, mul(g, u)))
        ghjb_rhs = scale(-1.0, add(l, u_sq))

        A = np.zeros((LEN_INDICES+1, LEN_INDICES+1))
        b = np.zeros(LEN_INDICES+1)
        for row,idx in enumerate(indices):
            b[row] = ghjb_rhs[idx]
            coefs = ghjb_lhs[idx].as_coefficients_dict()
            for col,idx_ in enumerate(indices):
                A[row,col] = coefs[v[idx_]]

        border_condition = c1d.evaluate(v, 0.0).as_coefficients_dict()
        for col, idx_ in enumerate(indices):
            A[-1,col] = border_condition[v[idx_]]
        lstsq_result = sl.lstsq(A, b)
        v_solved = c1d.of_dict({idx: lstsq_result[0][idx] for idx in indices})
        dv_solved = c1d.diff(v_solved)
        u = scale(-0.5, mul(g, dv_solved))

    print(u)
