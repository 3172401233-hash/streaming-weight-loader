import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from streaming_weight_loader import (
    ChecksumMismatchError,
    ShapeMismatchError,
    ShardRangeError,
    TruncatedShardError,
    UnknownTensorError,
    UnsupportedDTypeError,
    WeightLoader,
)
from streaming_weight_loader import dtypes


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class ShardFixture:
    """Builds shard files plus an index dict in a temp directory."""

    def __init__(self, case):
        self.case = case
        self.dir = tempfile.mkdtemp(prefix="swl-test-")
        case.addCleanup(shutil.rmtree, self.dir, True)
        self.tensors = {}
        self.blobs = {}  # shard -> bytearray

    def add_tensor(self, name, dtype, shape, values, shard="shard-0.bin",
                   byteorder="little", offset=None, checksum=True):
        raw = dtypes.encode(dtype, byteorder, values)
        blob = self.blobs.setdefault(shard, bytearray())
        if offset is None:
            offset = len(blob)
            blob += raw
        else:
            end = offset + len(raw)
            if end > len(blob):
                blob += b"\x00" * (end - len(blob))
            blob[offset:end] = raw
        entry = {
            "dtype": dtype,
            "shape": shape,
            "shard": shard,
            "offset": offset,
            "byteorder": byteorder,
        }
        if checksum:
            entry["sha256"] = sha256(raw)
        self.tensors[name] = entry
        return raw, offset

    def write(self, byteorder="little"):
        for shard, blob in self.blobs.items():
            with open(os.path.join(self.dir, shard), "wb") as fh:
                fh.write(bytes(blob))
        index = {"metadata": {"byteorder": byteorder}, "tensors": self.tensors}
        index_path = os.path.join(self.dir, "index.json")
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh)
        return index_path


class EndiannessTest(unittest.TestCase):
    def test_little_and_big_endian_roundtrip(self):
        fx = ShardFixture(self)
        floats = [1.5, -2.25, 3.75, 0.5]
        fx.add_tensor("le", "float32", [2, 2], floats, byteorder="little")
        fx.add_tensor("be", "float32", [2, 2], floats, byteorder="big")
        fx.add_tensor("be16", "int16", [4], [1, -2, 300, -300], byteorder="big")
        fx.add_tensor("bf16", "bfloat16", [2], [1.5, -0.25], byteorder="big")
        loader = WeightLoader(fx.write())

        self.assertEqual(loader.load("le").tolist(), [[1.5, -2.25], [3.75, 0.5]])
        self.assertEqual(loader.load("be").tolist(), [[1.5, -2.25], [3.75, 0.5]])
        self.assertEqual(loader.load("be16").values, [1, -2, 300, -300])
        self.assertEqual(loader.load("bf16").values, [1.5, -0.25])

    def test_same_bytes_decode_differently_per_byteorder(self):
        fx = ShardFixture(self)
        raw, offset = fx.add_tensor("le", "uint32", [1], [0x01020304],
                                    byteorder="little")
        # Reinterpret the identical bytes as big endian.
        fx.tensors["be"] = dict(fx.tensors["le"], byteorder="big",
                                sha256=sha256(raw))
        loader = WeightLoader(fx.write())
        self.assertEqual(loader.load("le").values, [0x01020304])
        self.assertEqual(loader.load("be").values, [0x04030201])


class OverlapTest(unittest.TestCase):
    def test_overlapping_tensors_share_bytes(self):
        fx = ShardFixture(self)
        values = [float(i) for i in range(8)]
        raw, offset = fx.add_tensor("full", "float32", [8], values)
        # A second tensor aliasing the middle of the first one's range.
        mid = raw[8:24]
        fx.tensors["view"] = {
            "dtype": "float32", "shape": [4], "shard": "shard-0.bin",
            "offset": offset + 8, "byteorder": "little", "sha256": sha256(mid),
        }
        loader = WeightLoader(fx.write())
        self.assertEqual(loader.load("full").values, values)
        self.assertEqual(loader.load("view").values, values[2:6])


class SparseReadTest(unittest.TestCase):
    def test_only_requested_ranges_are_read(self):
        fx = ShardFixture(self)
        fx.add_tensor("a", "float32", [4], [1.0] * 4)
        fx.add_tensor("b", "float32", [16], [2.0] * 16)
        fx.add_tensor("c", "float32", [64], [3.0] * 64)
        loader = WeightLoader(fx.write())

        tensor = loader.load("b")
        self.assertTrue(tensor.ready)
        self.assertEqual(loader.bytes_read, 16 * 4)

    def test_corrupt_neighbor_bytes_do_not_affect_sparse_load(self):
        fx = ShardFixture(self)
        fx.add_tensor("good", "float32", [2], [7.0, 8.0])
        fx.add_tensor("neighbor", "float32", [2], [1.0, 2.0])
        index_path = fx.write()
        # Trashing the neighbour's region must not break loading "good".
        path = os.path.join(fx.dir, "shard-0.bin")
        with open(path, "r+b") as fh:
            fh.seek(8)
            fh.write(b"\xff" * 8)
        loader = WeightLoader(index_path)
        self.assertEqual(loader.load("good").values, [7.0, 8.0])


class CorruptionTest(unittest.TestCase):
    def test_truncated_shard_raises(self):
        fx = ShardFixture(self)
        fx.add_tensor("w", "float32", [8], [1.0] * 8)
        index_path = fx.write()
        path = os.path.join(fx.dir, "shard-0.bin")
        with open(path, "r+b") as fh:
            fh.truncate(10)  # keep fewer bytes than the tensor needs
        loader = WeightLoader(index_path)
        with self.assertRaises(TruncatedShardError):
            loader.load("w")
        self.assertFalse(loader.get_tensor("w").ready)

    def test_offset_beyond_file_raises(self):
        fx = ShardFixture(self)
        raw, _ = fx.add_tensor("w", "float32", [2], [1.0, 2.0])
        fx.tensors["w"]["offset"] = 1 << 20  # corrupt offset far past EOF
        loader = WeightLoader(fx.write())
        with self.assertRaises(ShardRangeError):
            loader.load("w")

    def test_bad_checksum_raises_and_tensor_stays_not_ready(self):
        fx = ShardFixture(self)
        fx.add_tensor("w", "float32", [2], [1.0, 2.0])
        fx.tensors["w"]["sha256"] = "0" * 64
        loader = WeightLoader(fx.write())
        with self.assertRaises(ChecksumMismatchError):
            loader.load("w")
        tensor = loader.get_tensor("w")
        self.assertIsNotNone(tensor)
        self.assertFalse(tensor.ready)
        with self.assertRaises(RuntimeError):
            tensor.tolist()

    def test_shape_byte_mismatch_rejected(self):
        fx = ShardFixture(self)
        fx.add_tensor("w", "float32", [3], [1.0, 2.0, 3.0])
        fx.tensors["w"]["nbytes"] = 8  # 3 float32 == 12 bytes, not 8
        with self.assertRaises(ShapeMismatchError):
            WeightLoader(fx.write())

    def test_unknown_tensor_and_dtype(self):
        fx = ShardFixture(self)
        fx.add_tensor("w", "float32", [1], [1.0])
        loader = WeightLoader(fx.write())
        with self.assertRaises(UnknownTensorError):
            loader.load("nope")
        fx.tensors["bad"] = dict(fx.tensors["w"], dtype="float7")
        with self.assertRaises(UnsupportedDTypeError):
            WeightLoader(fx.write())


class ConcurrencyTest(unittest.TestCase):
    def test_concurrent_loads_all_ready_and_correct(self):
        fx = ShardFixture(self)
        expected = {}
        for i in range(40):
            shard = f"shard-{i % 4}.bin"
            values = [float(i)] * (i % 5 + 1)
            fx.add_tensor(f"t{i}", "float32", [len(values)], values, shard=shard)
            expected[f"t{i}"] = values
        loader = WeightLoader(fx.write())
        results = loader.load_many(loader.tensor_names(), max_workers=8)
        self.assertEqual(len(results), 40)
        for name, tensor in results.items():
            self.assertTrue(tensor.ready)
            self.assertEqual(tensor.values, expected[name])

    def test_tensor_not_ready_until_load_completes(self):
        fx = ShardFixture(self)
        fx.add_tensor("big", "float32", [256], [1.0] * 256)
        loader = WeightLoader(fx.write())

        release = threading.Event()
        original = loader._read_range

        def slow_read(shard, offset, nbytes):
            data = original(shard, offset, nbytes)
            release.wait(timeout=5)  # hold the load mid-flight
            return data

        loader._read_range = slow_read
        observed_not_ready = []

        def worker():
            loader.load("big")

        thread = threading.Thread(target=worker)
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            tensor = loader.get_tensor("big")
            if tensor is not None:
                observed_not_ready.append(not tensor.ready)
                break
            time.sleep(0.005)
        self.assertTrue(observed_not_ready and observed_not_ready[0],
                        "tensor was marked ready before its bytes were read")
        release.set()
        thread.join(timeout=5)
        self.assertTrue(loader.get_tensor("big").ready)


if __name__ == "__main__":
    unittest.main()
