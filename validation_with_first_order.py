import numpy as np
import scipy.linalg as sl
import cheby_tools.cheby1d as ch1d
from cheby_tools.cheby1d import add, sub, mul, scale, unary
import sympy as sym

def compute_LQR_gain(a, b, q, r):
    P = sl.solve_continuous_are(a, b, q, r)
    gain = r ** (-1.0) * b * P[0,0]
    return gain

PARAMETER_a = -1.0
PARAMETER_b = 3.0
PARAMETER_q = 3.0
PARAMETER_r = 1.0

def f_orig(x):
    return PARAMETER_a * x

def g_orig(x):
    return PARAMETER_b

def u_0_orig(x):
    # 極配置-1.0
    return  (-1.0 - PARAMETER_a) / PARAMETER_b * x

def l_orig(x):
    return PARAMETER_q * x ** 2.0

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
    N = 4
    indices = ch1d.generate_indices(N)
    LEN_INDICES = len(indices)

    # システムを構成する関数の多項式展開
    f = ch1d.expand(f_orig, indices)
    g = ch1d.expand(g_orig, indices)
    u_0 = ch1d.expand(u_0_orig, indices)

    # 評価関数を構成する関数の多項式展開
    l = ch1d.expand(l_orig, indices)

    # 未知変数v0...vNを係数に持つ多項式列
    v = ch1d.zero()
    for i in indices:
        new_symbol = sym.symbols("v_{}".format(i), real=True)
        v = add(v, unary(i, new_symbol))
    dv = ch1d.diff(v)

    u_log = []
    v_log = []

    u = u_0
    L = 20
    for i in range(L):
        # GHJB=0を計算．
        u_sq = ch1d.extract(scale(PARAMETER_r, mul(u, u)), indices)
        ghjb = add(mul(dv, add(f, mul(g, u))), add(l, u_sq))

        # v係数を求める方法を複数用意した．
        #   SOLVE
        #           V(0) = 0境界条件とK-1個の係数方程式からK個の方程式系を使って，解を求める．
        #           境界条件を満たすがかなり不安定．Nが奇数（項数が偶数）だと特異で計算不能．
        #   LSTSQ_WITHOUT_BOUNDARY_CONDITION
        #           K個の係数方程式から最小二乗解を求める．
        #           比較的安定しているがV(0) = 0境界条件を満たさない．
        #   LSTSQ_WITH_BOUNDARY_CONDITION
        #           K個の係数方程式にV(0) = 0境界条件を加えて，最小二乗解を求める．
        #           比較的安定していてV(0) = 0境界条件も満たすっぽい．
        SOLVE_MODE = "LSTSQ_WITH_BOUNDARY_CONDITION"
        if SOLVE_MODE == "SOLVE":
            # 境界条件を考慮する計算．
            C = np.zeros((LEN_INDICES + 1, LEN_INDICES + 1))
            y = np.zeros(LEN_INDICES + 1)

            # 最後の項ΦNについての内積は連立方程式に用いない．
            for row, i in enumerate(indices[:-1]):
                coefs = ghjb[i].as_coefficients_dict()
                y[row] = -coefs[1]
                for col, v_j in enumerate(v.values()):
                    C[row,col] = coefs[v_j]

            # ΦNについての内積の代わりに，V(0)=0を条件に用いる．
            v_boundary_coefs = ch1d.evaluate(v, 0.0).as_coefficients_dict()
            for col, v_j in enumerate(v.values()):
                C[LEN_INDICES,col] = v_boundary_coefs[v_j]

            v_coefs = sl.solve(C, y)
        elif SOLVE_MODE == "LSTSQ_WITHOUT_BOUNDARY_CONDITION":
            C = np.zeros((LEN_INDICES + 1, LEN_INDICES + 1))
            y = np.zeros(LEN_INDICES + 1)

            # C[:,0] == 0なのでCは必ず特異行列．
            for row, i in enumerate(indices):
                coefs = ghjb[i].as_coefficients_dict()
                y[row] = -coefs[1]
                for col, v_j in enumerate(v.values()):
                    C[row,col] = coefs[v_j]

            # 無理やり最小二乗法で計算する．
            v_coefs_result = sl.lstsq(C, y)
            v_coefs = v_coefs_result[0]
        elif SOLVE_MODE == "LSTSQ_WITH_BOUNDARY_CONDITION":
            # 境界条件を考慮する計算．
            C = np.zeros((LEN_INDICES + 2, LEN_INDICES + 1))
            y = np.zeros(LEN_INDICES + 2)

            for row, i in enumerate(indices):
                coefs = ghjb[i].as_coefficients_dict()
                y[row] = -coefs[1]
                for col, v_j in enumerate(v.values()):
                    C[row,col] = coefs[v_j]

            # V(0)=0を追加の方程式として導入．
            v_boundary_coefs = ch1d.evaluate(v, 0.0).as_coefficients_dict()
            for col, v_j in enumerate(v.values()):
                C[LEN_INDICES+1,col] = v_boundary_coefs[v_j]

            # 無理やり最小二乗法で計算する．
            v_coefs_result = sl.lstsq(C, y)
            v_coefs = v_coefs_result[0]
        else:
            raise NotImplementedError(f"SOLVE_MODE {SOLVE_MODE} is not implemented.")

        # 上で計算した係数から決定されたvを生成する．
        v_determined = ch1d.zero()
        for i, vc in zip(indices, v_coefs):
            v_determined = add(v_determined, unary(i, vc))
        dv_determined = ch1d.diff(v_determined)

        # u, vの記録を取る．
        u_log.append(u)
        v_log.append(v_determined)

        # 入力の更新．
        u_new = scale(-0.5 * PARAMETER_r ** (-1.0), mul(g, dv_determined))
        u = ch1d.extract(u_new, indices)

    print("Cheby result:", u_log[-1])
    print("LQR result:", compute_LQR_gain(PARAMETER_a, PARAMETER_b, PARAMETER_q, PARAMETER_r))
    print("V(0) approximation:", ch1d.evaluate(v_log[-1], 0.0))
