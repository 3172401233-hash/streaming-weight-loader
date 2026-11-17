"""Streaming, shard-aware weight loader.

Only the byte ranges belonging to requested tensors are read from disk;
shards are never mapped or loaded wholesale.  Every read is validated
against the shard bounds, the declared shape and the recorded checksum
before the tensor is marked ready.
"""

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from . import dtypes as _dtypes
from .errors import (
    ChecksumMismatchError,
    IndexFormatError,
    ShapeMismatchError,
    ShardRangeError,
    TruncatedShardError,
    UnknownTensorError,
)

_VALID_BYTEORDERS = ("little", "big")


def _num_elements(shape):
    n = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise IndexFormatError(f"invalid shape dimension: {dim!r}")
        n *= dim
    return n


def _normalize_entry(name, entry, default_byteorder):
    if not isinstance(entry, dict):
        raise IndexFormatError(f"tensor entry {name!r} must be an object")
    try:
        dtype = entry["dtype"]
        shape = list(entry["shape"])
        shard = entry["shard"]
        offset = entry["offset"]
    except KeyError as exc:
        raise IndexFormatError(
            f"tensor entry {name!r} is missing field {exc}"
        ) from None
    byteorder = entry.get("byteorder", default_byteorder)
    if byteorder not in _VALID_BYTEORDERS:
        raise IndexFormatError(
            f"tensor {name!r}: byteorder must be one of {_VALID_BYTEORDERS}"
        )
    if not isinstance(offset, int) or offset < 0:
        raise IndexFormatError(f"tensor {name!r}: invalid offset {offset!r}")
    if not isinstance(shard, str) or not shard:
        raise IndexFormatError(f"tensor {name!r}: invalid shard {shard!r}")

    itemsize = _dtypes.dtype_itemsize(dtype)  # raises UnsupportedDTypeError
    expected_nbytes = _num_elements(shape) * itemsize
    nbytes = entry.get("nbytes", expected_nbytes)
    if nbytes != expected_nbytes:
        raise ShapeMismatchError(
            f"tensor {name!r}: shape {shape} of {dtype} implies "
            f"{expected_nbytes} bytes, but index declares {nbytes}"
        )
    checksum = entry.get("sha256")
    if checksum is not None and not isinstance(checksum, str):
        raise IndexFormatError(f"tensor {name!r}: sha256 must be a hex string")
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "shard": shard,
        "offset": offset,
        "nbytes": nbytes,
        "byteorder": byteorder,
        "sha256": checksum,
    }


def load_index(source):
    """Parse and validate an index from a path, file object, or dict.

    Returns ``(metadata, tensors)`` where ``tensors`` maps names to
    normalized entries.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    elif hasattr(source, "read"):
        raw = json.load(source)
    elif isinstance(source, dict):
        raw = source
    else:
        raise IndexFormatError(f"cannot load index from {type(source).__name__}")

    if not isinstance(raw, dict) or "tensors" not in raw:
        raise IndexFormatError("index must be an object with a 'tensors' key")
    metadata = raw.get("metadata", {})
    default_byteorder = metadata.get("byteorder", "little")
    if default_byteorder not in _VALID_BYTEORDERS:
        raise IndexFormatError(
            f"metadata byteorder must be one of {_VALID_BYTEORDERS}"
        )
    tensors_raw = raw["tensors"]
    if isinstance(tensors_raw, list):  # allow a list of {"name": ..., ...}
        tensors_raw = {e["name"]: {k: v for k, v in e.items() if k != "name"}
                       for e in tensors_raw}
    if not isinstance(tensors_raw, dict):
        raise IndexFormatError("'tensors' must be an object or a list")
    tensors = {
        name: _normalize_entry(name, entry, default_byteorder)
        for name, entry in tensors_raw.items()
    }
    return metadata, tensors


class Tensor:
    """A tensor materialized from a shard byte range.

    ``ready`` is False until every byte has been read and validated;
    consumers must not use ``values`` before then.
    """

    __slots__ = ("name", "dtype", "shape", "byteorder", "_values", "_ready")

    def __init__(self, name, dtype, shape, byteorder):
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)
        self.byteorder = byteorder
        self._values = None
        self._ready = False

    @property
    def ready(self):
        return self._ready

    @property
    def values(self):
        """Flat list of decoded values (only meaningful when ready)."""
        return self._values

    def num_elements(self):
        n = 1
        for dim in self.shape:
            n *= dim
        return n

    def tolist(self):
        """Values reshaped into nested lists matching ``shape``."""
        if not self._ready:
            raise RuntimeError(f"tensor {self.name!r} is not ready")
        return _reshape(self._values, self.shape)

    def __repr__(self):
        state = "ready" if self._ready else "pending"
        return (
            f"Tensor(name={self.name!r}, dtype={self.dtype!r}, "
            f"shape={self.shape}, {state})"
        )


def _reshape(flat, shape):
    if not shape:
        return flat[0] if flat else None
    if len(shape) == 1:
        return list(flat)
    step = 1
    for dim in shape[1:]:
        step *= dim
    return [_reshape(flat[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


class WeightLoader:
    """Loads tensors on demand from sharded checkpoint files.

    Parameters
    ----------
    index:
        Path to a JSON index, a file object, or an already-parsed dict.
    shard_dir:
        Directory containing the shard files referenced by the index.
        Defaults to the index file's directory (or cwd for dict input).
    """

    def __init__(self, index, shard_dir=None):
        if shard_dir is None and isinstance(index, (str, os.PathLike)):
            shard_dir = os.path.dirname(os.path.abspath(index))
        self._shard_dir = os.path.abspath(shard_dir or ".")
        self.metadata, self._entries = load_index(index)
        self._shard_locks = {}
        self._shard_locks_guard = threading.Lock()
        self._tensors = {}
        self._tensors_guard = threading.Lock()
        self.bytes_read = 0  # diagnostic: total bytes fetched from shards
        self._bytes_guard = threading.Lock()

    # -- introspection -------------------------------------------------

    def tensor_names(self):
        return sorted(self._entries)

    def entry(self, name):
        try:
            return self._entries[name]
        except KeyError:
            raise UnknownTensorError(f"unknown tensor: {name!r}") from None

    def get_tensor(self, name):
        """Return the Tensor object for ``name`` if loading has begun."""
        with self._tensors_guard:
            return self._tensors.get(name)

    # -- shard I/O -----------------------------------------------------

    def _shard_path(self, shard):
        path = os.path.join(self._shard_dir, shard)
        # The shard must stay inside shard_dir (no path escape).
        if not os.path.abspath(path).startswith(self._shard_dir + os.sep):
            raise ShardRangeError(f"shard {shard!r} escapes shard directory")
        return path

    def _shard_lock(self, shard):
        with self._shard_locks_guard:
            return self._shard_locks.setdefault(shard, threading.Lock())

    def _read_range(self, shard, offset, nbytes):
        """Read exactly ``nbytes`` at ``offset`` from ``shard``.

        The range is checked against the actual file size so a corrupt
        offset can never leak bytes from a neighbouring tensor or past
        end-of-file.
        """
        path = self._shard_path(shard)
        lock = self._shard_lock(shard)
        with lock:
            try:
                size = os.path.getsize(path)
            except OSError as exc:
                raise ShardRangeError(
                    f"shard {shard!r} is not readable: {exc}"
                ) from None
            if offset < 0 or offset > size:
                raise ShardRangeError(
                    f"offset {offset} is outside shard {shard!r} (size {size})"
                )
            if offset + nbytes > size:
                raise TruncatedShardError(
                    f"range [{offset}, {offset + nbytes}) exceeds shard "
                    f"{shard!r} size {size}; file may be truncated"
                )
            with open(path, "rb", buffering=0) as fh:
                fh.seek(offset)
                chunks = []
                remaining = nbytes
                while remaining > 0:
                    chunk = fh.read(remaining)
                    if not chunk:
                        raise TruncatedShardError(
                            f"shard {shard!r} ended early at offset "
                            f"{offset + nbytes - remaining}"
                        )
                    chunks.append(chunk)
                    remaining -= len(chunk)
        data = b"".join(chunks)
        with self._bytes_guard:
            self.bytes_read += len(data)
        return data

    # -- loading -------------------------------------------------------

    def load(self, name):
        """Load a single tensor, reading only its byte range."""
        entry = self.entry(name)
        tensor = Tensor(name, entry["dtype"], entry["shape"], entry["byteorder"])
        with self._tensors_guard:
            self._tensors[name] = tensor

        raw = self._read_range(entry["shard"], entry["offset"], entry["nbytes"])

        if len(raw) != entry["nbytes"]:
            raise TruncatedShardError(
                f"tensor {name!r}: expected {entry['nbytes']} bytes, "
                f"got {len(raw)}"
            )
        if entry["sha256"] is not None:
            digest = hashlib.sha256(raw).hexdigest()
            if digest != entry["sha256"].lower():
                raise ChecksumMismatchError(
                    f"tensor {name!r}: sha256 {digest} != index "
                    f"{entry['sha256'].lower()}"
                )
        values = _dtypes.decode(entry["dtype"], entry["byteorder"], raw)
        if len(values) != tensor.num_elements():
            raise ShapeMismatchError(
                f"tensor {name!r}: decoded {len(values)} elements, "
                f"shape {entry['shape']} requires {tensor.num_elements()}"
            )
        # Only now, after every check has passed, publish the tensor.
        tensor._values = values
        tensor._ready = True
        return tensor

    def load_many(self, names, max_workers=4):
        """Load tensors concurrently; returns ``{name: Tensor}``.

        Any failure re-raises after all submitted loads settle, and the
        failed tensor is never marked ready.
        """
        names = list(names)
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {name: pool.submit(self.load, name) for name in names}
            for name, fut in futures.items():
                results[name] = fut.result()
        return results
