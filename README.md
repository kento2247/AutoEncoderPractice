# AutoEncoderPractice

MNIST を使って、以下 3 種類の自己符号化モデルを試すための小さな実験用リポジトリです。

- AutoEncoder (AE)
- Variational AutoEncoder (VAE)
- Vector Quantized VAE (VQ-VAE)

実装の中心は [`autoencoder.ipynb`](/home/initial/AutoEncoderPractice/autoencoder.ipynb) にあり、PyTorch で最小構成の学習ループを動かせます。

## できること

- MNIST 学習データの自動ダウンロード
- `AE / VAE / VQ-VAE` の切り替え実行
- 3 epoch の簡易学習
- CUDA が使える環境では GPU 実行、使えなければ CPU 実行

## 構成

- [`autoencoder.ipynb`](/home/initial/AutoEncoderPractice/autoencoder.ipynb): モデル定義と学習コード
- [`pyproject.toml`](/home/initial/AutoEncoderPractice/pyproject.toml): 依存関係
- [`data/`](/home/initial/AutoEncoderPractice/data): MNIST データ保存先

## 前提

- Python 3.11 以上
- `uv`
- NVIDIA GPU を使う場合は CUDA 11.8 対応環境

このリポジトリの `torch` は `cu118` 向け index を使う設定です。CPU のみで使う場合は、環境に応じて PyTorch の依存関係設定を調整してください。

## セットアップ

```bash
uv sync
```

仮想環境を有効化する場合:

```bash
source .venv/bin/activate
```

## 実行方法

このリポジトリはノートブック実行を前提にしています。

```bash
uv run jupyter notebook autoencoder.ipynb
```

Jupyter が未導入なら、先に追加してください。

```bash
uv add jupyter
```

ノートブック末尾では次のようにモデルを指定して実行します。

```python
main("ae")
```

利用できる引数:

- `"ae"`
- `"vae"`
- `"vqvae"`

例えば VAE を試す場合:

```python
main("vae")
```

## 実装概要

### AE

- Encoder: `784 -> 256 -> 64`
- Decoder: `64 -> 256 -> 784`
- 損失: MSE

### VAE

- 潜在変数の平均 `mu` と分散 `logvar` を推定
- Reparameterization trick を使用
- 損失: 再構成誤差 + KL divergence

### VQ-VAE

- 64 次元潜在表現をベクトル量子化
- コードブックサイズは `K=512`
- 損失: 再構成誤差 + VQ loss

## データ

初回実行時に `./data` 配下へ MNIST が保存されます。すでに [`data/MNIST/raw`](/home/initial/AutoEncoderPractice/data/MNIST/raw) に生データがある場合は、それを再利用します。

## 補足

- 学習 epoch 数は現在 3 に固定されています
- バッチサイズは 128 です
- 学習ログは各 epoch の最後に loss を標準出力へ表示します
- 現状は可視化や評価指標の保存は入っていません

## 今後拡張しやすい点

- 学習結果の画像可視化
- モデル種別や epoch 数の引数化
- Notebook から `.py` スクリプトへの切り出し
- 再構成画像と潜在表現の比較
