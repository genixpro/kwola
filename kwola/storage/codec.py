"""Versioned MessagePack and Zstandard record codec."""

import hashlib
from collections.abc import Mapping
from typing import Any

import msgpack
import zstandard

MAGIC = b"KWDB"
CODEC_VERSION = 1
DIGEST_SIZE = 16


class CodecError(ValueError):
    pass


class BinaryCodec:
    def __init__(self, compression_level: int = 3) -> None:
        self._compressor = zstandard.ZstdCompressor(level=compression_level)
        self._decompressor = zstandard.ZstdDecompressor()

    def encode(self, value: Mapping[str, Any]) -> bytes:
        packed = msgpack.packb(dict(value), use_bin_type=True, strict_types=True)
        compressed = self._compressor.compress(packed)
        digest = hashlib.blake2b(compressed, digest_size=DIGEST_SIZE).digest()
        return MAGIC + bytes((CODEC_VERSION,)) + digest + compressed

    def decode(self, payload: bytes) -> dict[str, Any]:
        header_size = len(MAGIC) + 1 + DIGEST_SIZE
        if len(payload) < header_size or not payload.startswith(MAGIC):
            raise CodecError("record has an invalid codec header")
        version = payload[len(MAGIC)]
        if version != CODEC_VERSION:
            raise CodecError(f"unsupported record codec version: {version}")
        expected = payload[len(MAGIC) + 1 : header_size]
        compressed = payload[header_size:]
        actual = hashlib.blake2b(compressed, digest_size=DIGEST_SIZE).digest()
        if actual != expected:
            raise CodecError("record checksum mismatch")
        try:
            result = msgpack.unpackb(self._decompressor.decompress(compressed), raw=False)
        except (msgpack.UnpackException, zstandard.ZstdError) as error:
            raise CodecError("record payload is corrupt") from error
        if not isinstance(result, dict):
            raise CodecError("record payload is not a mapping")
        return result
