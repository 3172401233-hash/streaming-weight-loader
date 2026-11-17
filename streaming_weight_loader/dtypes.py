"""Data type table and byte <-> value decoding helpers.

The loader is pure Python (no numpy dependency).  Values are decoded
into plain Python ints/floats via :mod:`struct`.
"""

import struct

from .errors import UnsupportedDTypeError

# name -> (itemsize_in_bytes, struct_format_char or None, kind)
# kind is "struct" for formats struct understands directly, or
# "bfloat16" which needs manual bit manipulation.
_DTYPES = {
    "float64": (8, "d", "struct"),
    "float32": (4, "f", "struct"),
    "float16": (2, "e", "struct"),
    "bfloat16": (2, None, "bfloat16"),
    "int64": (8, "q", "struct"),
    "int32": (4, "i", "struct"),
    "int16": (2, "h", "struct"),
    "int8": (1, "b", "struct"),
    "uint64": (8, "Q", "struct"),
    "uint32": (4, "I", "struct"),
    "uint16": (2, "H", "struct"),
    "uint8": (1, "B", "struct"),
    "bool": (1, "?", "struct"),
}

_BYTEORDER_PREFIX = {"little": "<", "big": ">"}


def supported_dtypes():
    return sorted(_DTYPES)


def dtype_itemsize(dtype):
    try:
        return _DTYPES[dtype][0]
    except KeyError:
        raise UnsupportedDTypeError(f"unsupported dtype: {dtype!r}") from None


def _check_dtype(dtype):
    if dtype not in _DTYPES:
        raise UnsupportedDTypeError(f"unsupported dtype: {dtype!r}")


def decode(dtype, byteorder, raw):
    """Decode ``raw`` bytes into a flat list of Python values."""
    _check_dtype(dtype)
    size, fmt, kind = _DTYPES[dtype]
    if len(raw) % size != 0:
        raise ValueError(
            f"byte length {len(raw)} is not a multiple of itemsize {size}"
        )
    count = len(raw) // size
    if count == 0:
        return []
    if kind == "struct":
        prefix = _BYTEORDER_PREFIX[byteorder]
        return list(struct.unpack(f"{prefix}{count}{fmt}", raw))
    # bfloat16: expand the upper 16 bits of float32
    values = []
    if byteorder == "little":
        for i in range(0, len(raw), 2):
            bits32 = raw[i] | (raw[i + 1] << 8)
            values.append(struct.unpack("<f", struct.pack("<I", bits32 << 16))[0])
    else:
        for i in range(0, len(raw), 2):
            bits32 = (raw[i] << 8) | raw[i + 1]
            values.append(struct.unpack(">f", struct.pack(">I", bits32 << 16))[0])
    return values


def encode(dtype, byteorder, values):
    """Encode a flat list of Python values into bytes (test/helper use)."""
    _check_dtype(dtype)
    _, fmt, kind = _DTYPES[dtype]
    if kind == "struct":
        prefix = _BYTEORDER_PREFIX[byteorder]
        return struct.pack(f"{prefix}{len(values)}{fmt}", *values)
    out = bytearray()
    for v in values:
        bits32 = struct.unpack("<I", struct.pack("<f", v))[0] >> 16
        if byteorder == "little":
            out += struct.pack("<H", bits32)
        else:
            out += struct.pack(">H", bits32)
    return bytes(out)
