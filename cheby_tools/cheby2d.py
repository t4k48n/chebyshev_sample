import sympy
import numpy as np
import numpy.fft as fft
from scipy.special import eval_chebyt
from scipy.integrate import dblquad
from collections import defaultdict
import sympy as sym

# これより小さいものはゼロとみなす．
ZERO_THRESHOLD = 1E-10
NUMBER_TYPES = (float, int, np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64)

def _is_zero(x):
    """ xが0近くの値であるか判定する．xがsympy.Expr型の場合は判定できないので，
    Falseを返す．値の判定にZERO_THRESHOLDグローバル変数を使う．
    引数:
        x   判定する値．数値（整数，浮動小数）かsympyの値を想定している．

    返り値:
        True or False
            浮動小数が0近い場合はTrueを返す．その他はFalseを返す．
    """
    return isinstance(x, NUMBER_TYPES) and abs(x) < ZERO_THRESHOLD

def of_dict(normal_dict):
    """ キーをインデックス(i,j)，バリューを係数cとする通常の辞書から，
    チェビシェフ辞書を生成する．

    引数:
        normal_dict キーをインデックス(i,j)，バリューを係数cとする通常の辞書

    返り値:
        チェビシェフ辞書（実体はdefaultdict）
    """
    return defaultdict(float, normal_dict)

def zero():
    """ 0とみなせるチェビシェフ辞書を生成する．

    返り値:
        0とみなせるチェビシェフ辞書．実体はdefaultdict(float, {})
    """
    return of_dict({})

def unary(i,j,c):
    """ チェビシェフの単項，c * Φ_i(x1) Φ_j(x2)を生成する．

    引数:
        i   x1に対応するチェビシェフ多項式のインデックス．
        j   x2に対応するチェビシェフ多項式のインデックス．
        c   項の係数．floatか，sympyの値を想定．
    """
    if _is_zero(c):
        return zero()
    return of_dict({(i,j): c})

def extract(cheby2d, indices):
    """ cheby2dからindicesのインデックスに対応する項だけ抽出して多項式列を作る．

    引数:
        cheby2d 抽出元の多項式列
        indices 抽出する項のインデックス
    """
    return of_dict({i: cheby2d[i] for i in indices if not _is_zero(cheby2d[i])})

def iadd(cheby2d_l, cheby2d_r):
    """ 左のチェビシェフ辞書に右のを破壊的に足し合わせる．

    引数:
        cheby2d_l   和の左辺になるチェビシェフ辞書．呼び出し後，値が変更される．
        cheby2d_r   和の右辺になるチェビシェフ辞書．

    返り値:
        和の結果であるチェビシェフ辞書cheby2d_l．
    """
    for k, v in cheby2d_r.items():
        cheby2d_l[k] += v
        if _is_zero(cheby2d_l[k]):
            del cheby2d_l[k]
    return cheby2d_l

def add(cheby2d_l, cheby2d_r):
    """ チェビシェフ辞書を足し合わせる．

    引数:
        cheby2d_l   和の左辺になるチェビシェフ辞書．
        cheby2d_r   和の右辺になるチェビシェフ辞書．

    返り値:
        和の結果であるチェビシェフ辞書．
    """
    return iadd(cheby2d_l.copy(), cheby2d_r)

def isub(cheby2d_l, cheby2d_r):
    """ 左のチェビシェフ辞書から右のを破壊的に引き算する．

    引数:
        cheby2d_l   差の左辺になるチェビシェフ辞書．呼び出し後，値が変更される．
        cheby2d_r   差の右辺になるチェビシェフ辞書．

    返り値:
        差の結果であるチェビシェフ辞書cheby2d_l．
    """
    for k, v in cheby2d_r.items():
        cheby2d_l[k] -= v
        if _is_zero(cheby2d_l[k]):
            del cheby2d_l[k]
    return cheby2d_l

def sub(cheby2d_l, cheby2d_r):
    """ チェビシェフ辞書を引き算する．

    引数:
        cheby2d_l   差の左辺になるチェビシェフ辞書．
        cheby2d_r   差の右辺になるチェビシェフ辞書．

    返り値:
        差の結果であるチェビシェフ辞書．
    """
    return isub(cheby2d_l.copy(), cheby2d_r)

def mul(cheby2d_l, cheby2d_r):
    """ チェビシェフ辞書を掛け算する．

    引数:
        cheby2d_l   積の左辺になるチェビシェフ辞書．
        cheby2d_r   積の右辺になるチェビシェフ辞書．

    返り値:
        積の結果であるチェビシェフ辞書．
    """
    ans = defaultdict(float, {})
    for (i,j), vl in cheby2d_l.items():
        for (k,l), vr in cheby2d_r.items():
            c = vl * vr / 4
            if _is_zero(c):
                continue
            ans[(i+k,j+l)] += c
            ans[(i+k,abs(j-l))] += c
            ans[(abs(i-k),j+l)] += c
            ans[(abs(i-k),abs(j-l))] += c
    return ans

def iscale(coefficient, cheby2d):
    """ チェビシェフ辞書をスカラ値で破壊的にスケーリング（スカラ倍）する．

    引数:
        coefficient スケーリング係数
        cheby2d     スケール対象のチェビシェフ辞書．呼び出し後，値が変更される．

    返り値:
        スケールされたチェビシェフ辞書cheby2d
    """
    if _is_zero(coefficient):
        cheby2d.clear()
        return cheby2d
    for k in cheby2d:
        cheby2d[k] *= coefficient
    return cheby2d

def scale(coefficient, cheby2d):
    """ チェビシェフ辞書をスカラ値でスケーリング（スカラ倍）する．

    引数:
        coefficient スケーリング係数
        cheby2d     スケール対象のチェビシェフ辞書

    返り値:
        スケールされたチェビシェフ辞書
    """
    return iscale(coefficient, cheby2d.copy())

def evaluate(cheby2d, x1, x2):
    """ チェビシェフ辞書にx1とx2の値を代入した結果を評価する．

    引数:
        cheby2d 評価対象のチェビシェフ辞書．
        x1      x1の値．floatを想定（多分sympy値でも動く）．
        x2      x2の値．floatを想定（多分sympy値でも動く）．

    返り値:
        floatまたはsympy値型の評価結果．
    """
    ans = 0.0
    for (i,j), v in cheby2d.items():
        ans += eval_chebyt(i,x1) * eval_chebyt(j,x2) * v
    return ans

def diff1(cheby2d):
    """ c Φ_i(x1) Φ_j(x2) + ... を表すチェビシェフ辞書をx1で微分した
    導関数を求める．
    内部でcheby1d.diff_unaryを利用しているので，効率はこれに依存する．

    引数:
        cheby2d チェビシェフ辞書

    返り値:
        cheby2dのx1に関する導関数．
    """
    from cheby_tools.cheby1d import diff_unary
    ans = zero()
    for (i, j), v in cheby2d.items():
        diff_cheby1d = diff_unary(i)
        for i_, v_ in diff_cheby1d.items():
            ans[(i_, j)] += v * v_
    return ans

def diff2(cheby2d):
    """ c Φ_i(x1) Φ_j(x2) + ... を表すチェビシェフ辞書をx2で微分した
    導関数を求める．
    内部でcheby1d.diffを利用しているので，効率はこれに依存する．

    引数:
        cheby2d チェビシェフ辞書

    返り値:
        cheby2dのx2に関する導関数．
    """
    from cheby_tools.cheby1d import diff_unary
    ans = zero()
    for (i, j), v in cheby2d.items():
        diff_cheby1d = diff_unary(j)
        for j_, v_ in diff_cheby1d.items():
            ans[(i, j_)] += v * v_
    return ans

def generate_indices(n):
    """ 0からnまでの2次元インデックスを生成する．一番小さいインデックスが
    (0, 0)で，一番大きいインデックスが(n, n)になる．合計で(n + 1)*(n + 1)個の
    インデックスになる．

    引数:
        n   インデックスに含まれる最大の整数．

    返り値:
        生成されたインデックス(i,j) (0 <= i,j <= n)からなるリスト．
    """
    indices = []
    for i in range(n+1):
        for j in range(n+1):
            indices.append((i,j))
    return indices

def expand_old(f, indices):
    """
    関数fをインデックス列indicesを基底としたchebyshev多項式列で展開する．
    ガラーキン法を使う．直接数値積分をする古い方法（廃止予定）．

    引数:
        f       f(x1,x2)の形式の関数
        indices indices=[(i,j),...]の形式のインデックスリスト
    返り値:
        defaultdict(float, {(i,j): c, ...})
            cはi,jをインデックスとした基底に対応する係数．
    """
    ans = zero()
    for (i,j) in indices:
        # 被積分関数をf(x1, x2)としたとき，x1=cos t, x2=cos tとおいたtについての関数
        # F(t1, t2) = f(cos t1, cos t2)を用いた次の内積，
        #   (2/π)^{2} ∫_{0}^{π} ∫_{0}^{π} F(t1, t2) cos(i t1) cos(j t2) dt1 dt2,
        # は次のチェビシェフ多項式の内積に等しい．
        #   (2/π)^{2} ∫_{-1}^{1} ∫_{-1}^{1} f(x1, x2) Ti(x1) Tj(x2) w(x1) w(x2) dx1 dx2,
        # ここでTn(x)はチェビシェフ多項式のn項目，w(x) = 1/√(1-x*x)は内積の重み関数である．
        # ただし，i=0やj=0では2で割る．両方0なら4で割る．
        integrated_func = lambda t1, t2: f(np.cos(t1), np.cos(t2)) * np.cos(i*t1) * np.cos(j*t2)
        integration, _ = dblquad(integrated_func, 0.0, np.pi, 0.0, np.pi)
        integration *= (2.0 / np.pi) ** 2.0
        if i == 0:
            integration /= 2.0
        if j == 0:
            integration /= 2.0
        ans[(i,j)] = integration
    return ans

def _next_pow2(n):
    """ n以上の2のべき乗を返す.
    引数:
        n   0以上の整数．
    返り値:
        p   n以上の2のべき乗．
    例:
        _next_pow2(1) == 2
        _next_pow2(0) == 1
        _next_pow2(550) == 1024
    """
    p = 1
    while p < n:
        p *= 2
    return p

def expand(f, indices):
    """
    関数fをインデックス列indicesを基底としたchebyshev多項式列で展開する．
    ガラーキン法に基づきFFTを行う．

    引数:
        f       f(x1,x2)の形式の関数
        indices indices=[(i,j),...]の形式のインデックスリスト
    返り値:
        defaultdict(float, {(i,j): c, ...})
            cはi,jをインデックスとした基底に対応する係数．
    """
    # サンプリング定理より展開項数の2倍以上のサンプルが必要．さらにFFTの効率よ
    # り，サンプル数に2のべき乗を使う．
    M = _next_pow2(max(i for i, _ in indices)*2 + 1)
    N = _next_pow2(max(j for _, j in indices)*2 + 1)
    m = np.arange(M)
    n = np.arange(N)
    # 1のN乗根をサンプル点に使う．
    x1 = 2*np.pi*m/M
    x2 = 2*np.pi*n/N
    X1, X2 = np.meshgrid(x1, x2)
    fv = np.vectorize(f)
    # x1, x2の軸方向とfftの分解方向を合わせるために転置する．
    data = fv(np.cos(X1), np.cos(X2)).T
    result = fft.fft2(data)
    result = result.real / M / N
    result[1:,:] *= 2.0
    result[:,1:] *= 2.0
    return of_dict({i: result[i] for i in indices if not _is_zero(result[i])})

if __name__ == "__main__":
    N = 3
    indices = generate_indices(N)

    # システムを構成する関数の多項式展開
    f1_orig = lambda x1, x2: 2.0 * x1**2 - 1.0
    #f1_orig = lambda x1, x2: 3.14
    f1 = expand(f1_orig, indices)
    print(f1)
    f1 = expand_old(f1_orig, indices)
    print(f1)
