# streaming-weight-loader

Python 模型权重流式加载库：按索引从分片（shard）文件中**只读取目标张量的字节范围**，
避免一次性映射全部文件造成的内存峰值。所有读取都经过范围、形状和校验三重验证，
加载完成前张量不会被标记为 ready。纯标准库实现，无第三方依赖。

## 用法

```python
from streaming_weight_loader import WeightLoader

loader = WeightLoader("model/index.json", shard_dir="model/")
tensor = loader.load("layer.0.weight")   # 只读取该张量的字节范围
tensor.ready                             # True（校验全部通过后才会置位）
tensor.tolist()                          # 按 shape 嵌套的值

# 并发加载多个张量
tensors = loader.load_many(loader.tensor_names(), max_workers=8)
```

## 索引格式

JSON 文件，包含可选的 `metadata` 和必需的 `tensors`：

```json
{
  "metadata": {"byteorder": "little"},
  "tensors": {
    "layer.0.weight": {
      "dtype": "float32",
      "shape": [768, 768],
      "shard": "model-00001.bin",
      "offset": 0,
      "byteorder": "big",
      "nbytes": 2359296,
      "sha256": "<hex digest>"
    }
  }
}
```

字段说明：

- `dtype`（必需）：数据类型，见下表。
- `shape`（必需）：维度列表；元素个数 × 元素字节数必须与字节长度一致，否则抛出
  `ShapeMismatchError`。
- `shard`（必需）：分片文件名（相对索引所在目录，禁止路径逃逸）。
- `offset`（必需）：张量数据在分片内的字节偏移，必须 ≥ 0 且
  `[offset, offset+nbytes)` 完整落在分片文件内，否则抛出 `ShardRangeError` /
  `TruncatedShardError`。
- `byteorder`（可选）：`"little"` 或 `"big"`，默认取 `metadata.byteorder`（缺省
  `"little"`）。
- `nbytes`（可选）：字节长度，缺省由 `shape` 和 `dtype` 推导；若显式给出则必须一致。
- `sha256`（可选）：张量字节范围的 SHA-256 十六进制摘要；不匹配时抛出
  `ChecksumMismatchError`，且张量保持 not-ready。

`tensors` 也接受列表形式（每项含 `name` 字段）。

## 支持的数据类型

| dtype      | 字节数 | 说明                     |
|------------|--------|--------------------------|
| `float64`  | 8      | 双精度浮点               |
| `float32`  | 4      | 单精度浮点               |
| `float16`  | 2      | 半精度浮点               |
| `bfloat16` | 2      | bfloat16（手动位扩展）   |
| `int64`    | 8      | 有符号整数               |
| `int32`    | 4      | 有符号整数               |
| `int16`    | 2      | 有符号整数               |
| `int8`     | 1      | 有符号整数               |
| `uint64`   | 8      | 无符号整数               |
| `uint32`   | 4      | 无符号整数               |
| `uint16`   | 2      | 无符号整数               |
| `uint8`    | 1      | 无符号整数               |
| `bool`     | 1      | 布尔                     |

所有整型/浮点类型均支持大端（`big`）与小端（`little`）字节序。

## 错误类型

`WeightLoaderError` 为基类，子类包括：`IndexFormatError`、`UnknownTensorError`、
`UnsupportedDTypeError`、`ShapeMismatchError`、`ShardRangeError`（子类
`TruncatedShardError`）、`ChecksumMismatchError`。

## 测试

```bash
python -m unittest
```

覆盖大端/小端、重叠张量、稀疏读取、截断文件、错误摘要、形状不一致以及并发加载。
