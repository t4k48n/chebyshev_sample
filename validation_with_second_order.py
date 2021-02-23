import numpy as np
import scipy.linalg as sl
import cheby_tools.cheby2d as ch2d
from cheby_tools.cheby2d import add, sub, mul, scale, unary
import sympy as sym

import matplotlib.pyplot as plt

def compute_LQR_gain(A, B, Q, R):
    P = sl.solve_continuous_are(A, B, Q, R)
    gain = sl.inv(R) @ B.T @ P
    return gain

PARAMETER_A = np.array([[ 0.0,  1.0],
                        [-9.8, -1.0]])
PARAMETER_B = np.array([[0.0],
                        [1.0]])
PARAMETER_Q = np.eye(2)
PARAMETER_R = np.eye(1)

def f1_orig(x1, x2):
    return PARAMETER_A[0,:] @ np.array([x1, x2])

def f2_orig(x1, x2):
    #return PARAMETER_A[1,:] @ np.array([x1, x2])
    return -9.8 * np.sin(x1) - 1.0 * x2

def g1_orig(x1, x2):
    return PARAMETER_B[0,0]

def g2_orig(x1, x2):
    return PARAMETER_B[1,0]

def u_0_orig(x1, x2):
    return 0.0

def l_orig(x1, x2):
    x = np.array([x1, x2])
    return x @ PARAMETER_Q @ x

def plot_log(log, indices):
    """ logを人間が見やすい形でプロットする．
    引数:
        log     [{i: c}]形式のデータ，ここでiはインデックス，cはiに対応する係数．
        indices プロットしたい係数に対応するインデックスからなるリスト．
    """
    data = np.array([[logline[i] for i in indices] for logline in log])
    for i, line in zip(indices, data.T):
        plt.plot(line, label=str(i))

if __name__ == "__main__":
    N = 9
    indices = ch2d.generate_indices(N)
    LEN_INDICES = len(indices)

    # システムを構成する関数の多項式展開
    f1 = ch2d.expand(f1_orig, indices)
    f2 = ch2d.expand(f2_orig, indices)
    g1 = ch2d.expand(g1_orig, indices)
    g2 = ch2d.expand(g2_orig, indices)
    u_0 = ch2d.expand(u_0_orig, indices)

    # 評価関数を構成する関数の多項式展開
    l = ch2d.expand(l_orig, indices)

    # 未知変数v0...vNを係数に持つ多項式列
    v = ch2d.zero()
    for i, j in indices:
        new_symbol = sym.symbols("v_{}_{}".format(i, j))
        v = add(v, unary(i, j, new_symbol))
    dv1 = ch2d.diff1(v)
    dv2 = ch2d.diff2(v)

    u_log = []
    v_log = []

    u = u_0
    L = 1
    for i in range(L):
        # GHJB=0を計算．
        u_sq = scale(PARAMETER_R[0,0], ch2d.extract(mul(u, u), indices))
        ghjb_lhs = add(ch2d.extract(mul(dv1, add(f1, ch2d.extract(mul(g1, u), indices))), indices),
                       ch2d.extract(mul(dv2, add(f2, ch2d.extract(mul(g2, u), indices))), indices))
        ghjb_rhs = scale(-1.0, add(l, u_sq))

        # v係数を求める方法を複数用意した．
        #   SOLVE
        #           V(0) = 0境界条件とK-1個の係数方程式からK個の方程式系を使って，解を求める．
        #           境界条件を満たすがかなり不安定．
        #   LSTSQ_WITHOUT_BOUNDARY_CONDITION
        #           K個の係数方程式から最小二乗解を求める．
        #           比較的安定しているがV(0) = 0境界条件を満たさない．
        #   LSTSQ_WITH_BOUNDARY_CONDITION
        #           K個の係数方程式にV(0) = 0境界条件を加えて，最小二乗解を求める．
        #           比較的安定していてV(0) = 0境界条件も満たすっぽい．
        SOLVE_MODE = "LSTSQ_WITH_BOUNDARY_CONDITION"
        if SOLVE_MODE == "SOLVE":
            # 境界条件を考慮する計算．
            C = np.zeros((LEN_INDICES, LEN_INDICES))
            y = np.zeros(LEN_INDICES)

            # 最後の項ΦNについての内積は連立方程式に用いない．
            for row, i in enumerate(indices[:-1]):
                coefs = ghjb_lhs[i].as_coefficients_dict()
                y[row] = ghjb_rhs[i]
                for col, (i_, j_) in enumerate(indices):
                    v_j = v[(i_,j_)]
                    C[row,col] = coefs[v_j]

            # ΦNについての内積の代わりに，V(0)=0を条件に用いる．
            v_boundary_coefs = ch2d.evaluate(v, 0.0, 0.0).as_coefficients_dict()
            for col, (i_,j_) in enumerate(indices):
                v_j = v[(i_,j_)]
                C[LEN_INDICES-1,col] = v_boundary_coefs[v_j]

            v_coefs = sl.solve(C, y)
        elif SOLVE_MODE == "LSTSQ_WITHOUT_BOUNDARY_CONDITION":
            C = np.zeros((LEN_INDICES, LEN_INDICES))
            y = np.zeros(LEN_INDICES)

            # C[:,0] == 0なのでCは必ず特異行列．
            for row, i in enumerate(indices):
                coefs = ghjb_lhs[i].as_coefficients_dict()
                y[row] = ghjb_rhs[i]
                for col, (i_,j_) in enumerate(indices):
                    v_j = v[(i_,j_)]
                    C[row,col] = coefs[v_j]

            # 無理やり最小二乗法で計算する．
            v_coefs_result = sl.lstsq(C, y)
            v_coefs = v_coefs_result[0]
        elif SOLVE_MODE == "LSTSQ_WITH_BOUNDARY_CONDITION":
            # 境界条件を考慮する計算．
            C = np.zeros((LEN_INDICES + 1, LEN_INDICES))
            y = np.zeros(LEN_INDICES + 1)

            for row, i in enumerate(indices):
                coefs = ghjb_lhs[i].as_coefficients_dict()
                y[row] = ghjb_rhs[i]
                for col, (i_,j_) in enumerate(indices):
                    C[row,col] = coefs[v[(i_,j_)]]

            # V(0)=0を追加の方程式として導入．
            v_boundary_coefs = ch2d.evaluate(v, 0.0, 0.0).as_coefficients_dict()
            for col, (i_,j_) in enumerate(indices):
                v_j = v[(i_,j_)]
                C[LEN_INDICES,col] = v_boundary_coefs[v_j]

            # 無理やり最小二乗法で計算する．
            v_coefs_result = sl.lstsq(C, y)
            v_coefs = v_coefs_result[0]
        else:
            raise NotImplementedError(f"SOLVE_MODE {SOLVE_MODE} is not implemented.")

        # 上で計算した係数から決定されたvを生成する．
        v_determined = ch2d.zero()
        for (i, j), vc in zip(indices, v_coefs):
            v_determined = add(v_determined, unary(i, j, vc))
        dv1_determined = ch2d.diff1(v_determined)
        dv2_determined = ch2d.diff2(v_determined)

        # u, vの記録を取る．
        u_log.append(u)
        v_log.append(v_determined)

        # 入力の更新．
        u_new = scale(-0.5 * PARAMETER_R[0,0] ** (-1.0), add(mul(g1, dv1_determined), mul(g2, dv2_determined)))
        u = ch2d.extract(u_new, indices)

    print("Cheby result:", u_log[-1])
    print("LQR result:", compute_LQR_gain(PARAMETER_A, PARAMETER_B, PARAMETER_Q, PARAMETER_R))
    print("V(0) approximation:", ch2d.evaluate(v_log[-1], 0.0, 0.0))
