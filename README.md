# cheby\_toolsパッケージ

1次元および2次元のチェビシェフ多項式展開をあつかうパッケージ．

## パッケージ構成

- cheby\_tools/
    - \_\_init\_\_.py
    - cheby1d.py
    - cheby2d.py

`__init__.py`はcheby\_tools全体をパッケージとして認識させるために設置したからファイルだ．

## 開発環境

このコードを動かしている環境は`conda_env.txt`に保存している．次のコマンドで環境を復元することができる．

```
cond create -n <env_name> --file conda_env.txt
```

これで復元できない場合，ファイルを参照して手動インストールするほかない．環境を構築するために能動的にインストールしたパッケージは次のとおりだ．

- numpy
- scipy
- sympy
- matplotlib
- ipython

このうちipythonは対話環境を便利にするために入れた．最小環境には必要ない．numpy, scipyはバックエンドにmklを使っているが，openblas版でも問題はない．

## TODO

- 多項式乗算をFFTで実装できるか検討する．
- 係数の計算の実装をsympyに依存しない形に変える．

## 最適化にかんするメモ

2次元，N = 9 + 1 = 10で，f2が非線形関数-9.8 sin(x1) - 1.0 x2のシミュレーションは重い．どうしても項の数が増えて，sympy.Symbolの計算が多数登場してしまう．抽象的な計算はとても計算コストが高い．

1次元の場合は，ある程度の項数で安定してくれるので問題は表面化しなかった．1次元の場合，ループ回数Lとループ全体の計算時間の関係は次のようになる．

```
L = 1 -> 5.52 ms
L = 2 -> 8.67 ms
L = 10 -> 692 ms
L = 20 -> 1.34 s
L = 50 -> 3.41 s
```

1次元の場合は次のようになる．

```
L = 1 -> 1.86 s
L = 2 -> 27.2 s
```

2次元はあまりにも時間がかかりすぎている．解決方法は，sympyを使わないという方法以外にないだろう．
