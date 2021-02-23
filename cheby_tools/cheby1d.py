import sympy
import numpy as np
import numpy.fft as fft
from scipy.special import eval_chebyt
from scipy.integrate import quad
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
    """ キーをインデックスi，バリューを係数cとする通常の辞書から，
    チェビシェフ辞書を生成する．

    引数:
        normal_dict キーをインデックスi，バリューを係数cとする通常の辞書

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

def unary(i, c):
    """ チェビシェフの単項，c * Φ_i(x)を生成する．

    引数:
        i   x1に対応するチェビシェフ多項式のインデックス．
        c   項の係数．floatか，sympyの値を想定．

    返り値:
        c * Φ_i(x)を表すチェビシェフ辞書．
    """
    if _is_zero(c):
        return zero()
    return of_dict({i: c})

def extract(cheby1d, indices):
    """ cheby1dからindicesのインデックスに対応する項だけ抽出して多項式列を作る．

    引数:
        cheby1d 抽出元の多項式列
        indices 抽出する項のインデックス
    """
    return of_dict({i: cheby1d[i] for i in indices if not _is_zero(cheby1d[i])})

def iadd(cheby1d_l, cheby1d_r):
    """ 左のチェビシェフ辞書に右のものを破壊的に足し合わせる．

    引数:
        cheby1d_l   和の左辺になるチェビシェフ辞書．呼び出し後に値が変化する．
        cheby1d_r   和の右辺になるチェビシェフ辞書．

    返り値:
        加算の結果が格納された，左のチェビシェフ辞書．
    """
    for k, v in cheby1d_r.items():
        cheby1d_l[k] += v
        if _is_zero(cheby1d_l[k]):
            del cheby1d_l[k]
    return cheby1d_l

def add(cheby1d_l, cheby1d_r):
    """ チェビシェフ辞書を足し合わせた結果を格納したコピーを返す．

    引数:
        cheby1d_l   和の左辺になるチェビシェフ辞書．
        cheby1d_r   和の右辺になるチェビシェフ辞書．

    返り値:
        和の結果であるチェビシェフ辞書．
    """
    cheby1d_l_ = cheby1d_l.copy()
    return iadd(cheby1d_l_, cheby1d_r)

def isub(cheby1d_l, cheby1d_r):
    """ 左のチェビシェフ辞書から右のものを破壊的に引く．

    引数:
        cheby1d_l   差の左辺になるチェビシェフ辞書．呼び出し後に値が変化する．
        cheby1d_r   差の右辺になるチェビシェフ辞書．

    返り値:
        減算の結果が格納された，左のチェビシェフ辞書．
    """
    for k, v in cheby1d_r.items():
        cheby1d_l[k] -= v
        if _is_zero(cheby1d_l[k]):
            del cheby1d_l[k]
    return cheby1d_l

def sub(cheby1d_l, cheby1d_r):
    """ チェビシェフ辞書を引き算する．

    引数:
        cheby1d_l   差の左辺になるチェビシェフ辞書．
        cheby1d_r   差の右辺になるチェビシェフ辞書．

    返り値:
        差の結果であるチェビシェフ辞書．
    """
    cheby1d_l_ = cheby1d_l.copy()
    return isub(cheby1d_l_, cheby1d_r)

def mul(cheby1d_l, cheby1d_r):
    """ チェビシェフ辞書を掛け算する．

    引数:
        cheby1d_l   積の左辺になるチェビシェフ辞書．
        cheby1d_r   積の右辺になるチェビシェフ辞書．

    返り値:
        積の結果であるチェビシェフ辞書．
    """
    ans = zero()
    for i, vl in cheby1d_l.items():
        for j, vr in cheby1d_r.items():
            c = vl * vr / 2
            if _is_zero(c):
                continue
            ans[i+j] += c
            ans[abs(i-j)] += c
    return ans

def iscale(coefficient, cheby1d):
    """ チェビシェフ辞書をスカラ値で破壊的にスケーリング（スカラ倍）する．

    引数:
        coefficient スケーリング係数
        cheby1d     スケール対象のチェビシェフ辞書．
                    スケーリング後，値が変化する．

    返り値:
        スケール変更された後のcheby1d．
    """
    if _is_zero(coefficient):
        cheby1d.clear()
        return cheby1d
    for k in cheby1d:
        cheby1d[k] *= coefficient
    return cheby1d

def scale(coefficient, cheby1d):
    """ チェビシェフ辞書をスカラ値でスケーリング（スカラ倍）する．

    引数:
        coefficient スケーリング係数
        cheby1d     スケール対象のチェビシェフ辞書

    返り値:
        スケールされたチェビシェフ辞書
    """
    return iscale(coefficient, cheby1d.copy())

def evaluate(cheby1d, x):
    """ チェビシェフ辞書にxの値を代入した結果を評価する．

    引数:
        cheby1d 評価対象のチェビシェフ辞書．
        x       xの値．floatを想定（多分sympy値でも動く）．

    返り値:
        floatまたはsympy値型の評価結果．
    """
    return sum(eval_chebyt(i, x) * v for i, v in cheby1d.items())

def diff_unary(n):
    """ チェビシェフ単項Φ_n(x)をxで微分した導関数を返す．
    再帰処理で書いているため，スタックオーバーフローする．
    また同じ処理が何度も走るため非効率．暫定的な実装．

    引数:
        n   微分対象のインデックス

    返り値:
        微分した導関数．
    """
    if n == 0:
        return zero()
    elif n == 1:
        return unary(0, 1)
    elif n == 2:
        return unary(1, 4)
    else:
        return iadd(iscale(1 + 2 / (n - 2), diff_unary(n - 2)), unary(n - 1, 2 * n))

def diff(cheby1d):
    """ c Φ_i(x) + ... を表すチェビシェフ辞書をxで微分した導関数を求める．
    内部でdiff_unaryを利用しているので，効率はこれに依存する．

    引数:
        cheby1d チェビシェフ辞書

    返り値:
        cheby1dのxに関する導関数．
    """
    ans = zero()
    for i, v in cheby1d.items():
        iadd(ans, scale(v, diff_unary(i)))
    return ans

def generate_indices(n):
    """ 0からnまでの1次元インデックスを生成する．一番小さいインデックスが
    0で，一番大きいインデックスがnになる．合計でn + 1個のインデックスになる．
    実体はlist(range(n + 1))だが，cheby2dとの互換性のために関数が作られている．

    引数:
        n   インデックスに含まれる最大の整数．

    返り値:
        生成されたインデックスi (0 <= i <= n)からなるリスト．
    """
    return list(range(n + 1))

def expand_old(f, indices):
    """
    関数fをインデックス列indicesを基底としたchebyshev多項式列で展開する．
    ガラーキン法を使う．直接数値積分をする古い方法（廃止予定）．

    引数:
        f       f(x)の形式の関数
        indices [i1, i2, ...]の形式のインデックスリスト
    返り値:
        defaultdict(float, {i1: c1, i2: c2, ...})
            c1, c2はそれぞれi1, i2をインデックスとした基底に対応する係数．
    """
    ans = zero()
    for i in indices:
        # 被積分関数をf(x)としたとき，x=cos tとおいたtについての関数
        # F(t) = f(cos t)を用いた次の内積，
        #   (2/π) ∫_{0}^{π} F(t) cos(i t) dt
        # は次のチェビシェフ多項式の内積に等しい．
        #   (2/π) ∫_{-1}^{1} f(x) Ti(x) w(x) dx,
        # ここでTn(x)はチェビシェフ多項式のn項目，w(x) = 1/√(1-x*x)は内積の重み関数である．
        # ただし，i=0では2で割る．
        integrated_func = lambda t: f(np.cos(t)) * np.cos(i*t)
        integration, _ = quad(integrated_func, 0.0, np.pi)
        integration *= 2.0 / np.pi
        if i == 0:
            integration /= 2.0
        ans[i] = integration
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
        f       f(x)の形式の関数
        indices [i1, i2, ...]の形式のインデックスリスト
    返り値:
        defaultdict(float, {i1: c1, i2: c2, ...})
            c1, c2はそれぞれi1, i2をインデックスとした基底に対応する係数．
    """
    # サンプリング定理より展開項数の2倍以上のサンプルが必要．さらにFFTの効率よ
    # り，サンプル数に2のべき乗を使う．
    N = _next_pow2(max(indices)*2 + 1)
    n = np.arange(N)
    # 1のN乗根をサンプル点に使う．
    x = 2*np.pi*n/N
    fv = np.vectorize(f)
    data = fv(np.cos(x))
    # FFTして，
    result = fft.fft(data)
    # cos要素を取り出し，
    result = result.real / N
    # 適切にスケーリング．
    result[1:] *= 2.0
    return of_dict({i: result[i] for i in indices if not _is_zero(result[i])})

if __name__ == "__main__":
    f_orig = lambda x: 2.0 * x**2.0 - 1.0 + 3.0 * x
    f = expand(f_orig, generate_indices(3))
    print(f)

    n = 10
    # 微分の関係式で求めた辞書の評価
    e1 = evaluate(diff_unary(n), 0.5)
    # 数値微分（中間微分）で求めた評価
    h = 0.00000001
    e2 = (eval_chebyt(n,0.5+h) - eval_chebyt(n,0.5-h)) / (2.0 * h)
    print(e1, e2)

    print(add(diff_unary(3), diff_unary(5)), diff(of_dict({3: 1.0, 5: 1.0})))

    f_orig = lambda x: 2.0 * x**2 - 1 + 3.14 - 2.72 * x**3

    indices = generate_indices(3)
    f = expand_old(f_orig, indices)
    f_new = expand(f_orig, indices)
    print(f)
    print(f_new)

    c = of_dict({0: 1.0, 2: 3.14})
    cs = scale(10.0, c)
    print(c, cs)
    c = of_dict({0: 1.0, 2: 3.14})
    cs = iscale(10.0, c)
    print(c, cs)
