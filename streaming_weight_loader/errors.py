"""Exception hierarchy for the streaming weight loader."""


class WeightLoaderError(Exception):
    """Base class for all loader errors."""


class IndexFormatError(WeightLoaderError):
    """The index file is malformed or internally inconsistent."""


class UnknownTensorError(WeightLoaderError):
    """Requested tensor name is not present in the index."""


class UnsupportedDTypeError(WeightLoaderError):
    """The index references a dtype this loader cannot decode."""


class ShapeMismatchError(WeightLoaderError):
    """Shape element count and byte length disagree."""


class ShardRangeError(WeightLoaderError):
    """The requested byte range falls outside the shard file."""


class TruncatedShardError(ShardRangeError):
    """The shard file ended before the requested range was fully read."""


class ChecksumMismatchError(WeightLoaderError):
    """The bytes read from the shard do not match the recorded digest."""
