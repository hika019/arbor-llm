"""MLX port of Arbor (static patching + BitNet b1.58).

PyTorch/MPS 実装 (src/model/arbor.py) の Apple Silicon 向け MLX 版。
目的は Mac 上での学習スループット比較。static patching のみ対応。
"""
