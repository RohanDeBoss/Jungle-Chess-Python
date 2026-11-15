"""Reader and conversion helpers for diagonal-compressed five-man tables."""

import json
import os
import struct

import numpy as np


MAGIC = b"JCDIAG01"
HEADER = struct.Struct("<8sI")
RANK_BLOCK_BITS = 4096
_BIT_COUNTS = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _transpose(square):
    return (square % 8) * 8 + square // 8


def _transpose_canonical_keep(wk, squares, same_pairs):
    """Match the generator's diagonal tie-break for a chunk of flat states."""
    transposed = [_transpose(sq) for sq in squares]
    for left, right in same_pairs:
        swap = transposed[left] > transposed[right]
        left_values = np.where(swap, transposed[right], transposed[left])
        right_values = np.where(swap, transposed[left], transposed[right])
        transposed[left], transposed[right] = left_values, right_values

    less = np.zeros(wk.size, dtype=bool)
    decided = np.zeros(wk.size, dtype=bool)
    for original, mirrored in zip(squares, transposed):
        less |= (~decided) & (original < mirrored)
        decided |= original != mirrored
    return less | ~decided


def _decode_local(local):
    rest = local >> 1
    bk = rest & 63
    rest >>= 6
    p3 = rest & 63
    rest >>= 6
    p2 = rest & 63
    rest >>= 6
    p1 = rest & 63
    return p1, p2, p3, bk


def _encode_local(p1, p2, p3, bk, turn):
    return (((((p1 * 64 + p2) * 64 + p3) * 64 + bk) << 1) | turn)


class DiagonalCompressedTable:
    """Scalar-indexed ndarray-like view over a packed non-pawn 5-man table."""

    def __init__(self, filename, shape, base_name):
        self.filename = filename
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.base_name = base_name
        self.map_filename = filename + ".diagmap"
        with open(self.map_filename, "rb") as handle:
            raw = handle.read(HEADER.size)
            magic, header_size = HEADER.unpack(raw)
            if magic != MAGIC:
                raise ValueError(f"Invalid diagonal map signature: {self.map_filename}")
            self.meta = json.loads(handle.read(header_size).decode("utf-8"))
        if tuple(self.meta["shape"]) != self.shape:
            raise ValueError(f"Diagonal map shape mismatch for {base_name}")
        self.dtype = np.dtype(self.meta["dtype"])
        self.full_count = int(self.meta["full_count"])
        self.compact_count = int(self.meta["compact_count"])
        self.block_size = int(self.meta["block_size"])
        self._flat = np.memmap(filename, dtype=self.dtype, mode="r", shape=(self.compact_count,))
        self._map = np.memmap(self.map_filename, dtype=np.uint8, mode="r")
        self._blocks = {int(item["wk_index"]): item for item in self.meta["blocks"]}
        self._removed_before = {}
        removed = 0
        for block_index in range(self.shape[0]):
            self._removed_before[block_index] = removed
            info = self._blocks.get(block_index)
            if info:
                removed += int(info["removed_count"])

    def _removed_before_local(self, info, local):
        segment = local // RANK_BLOCK_BITS
        segment_start = segment * RANK_BLOCK_BITS
        prefix_offset = self.meta["payload_offset"] + int(info["prefix_offset"])
        prefix = np.frombuffer(
            self._map[prefix_offset:prefix_offset + int(info["prefix_count"]) * 4],
            dtype="<u4",
        )
        count = int(prefix[segment])
        bit_offset = local - segment_start
        mask_offset = self.meta["payload_offset"] + int(info["mask_offset"])
        full_bytes, tail_bits = divmod(bit_offset, 8)
        if full_bytes:
            first = mask_offset + segment_start // 8
            packed = self._map[first:first + full_bytes]
            count += int(_BIT_COUNTS[packed].sum(dtype=np.uint64))
        if tail_bits:
            byte_index = mask_offset + segment_start // 8 + full_bytes
            count += (int(self._map[byte_index]) & ((1 << tail_bits) - 1)).bit_count()
        return count

    def flat_index(self, flat):
        flat = int(flat)
        if flat < 0 or flat >= self.full_count:
            raise IndexError(flat)
        wk_index, local = divmod(flat, self.block_size)
        info = self._blocks.get(wk_index)
        if info is None:
            return flat - self._removed_before[wk_index]

        mask_offset = self.meta["payload_offset"] + int(info["mask_offset"])
        byte_index, bit = divmod(local, 8)
        removed = (int(self._map[mask_offset + byte_index]) >> bit) & 1
        if removed:
            # Probe canonicalization should always select the retained alias.
            # Raise instead of silently returning an unrelated packed value.
            raise IndexError(f"Non-canonical diagonal alias requested in {self.base_name}")
        return (wk_index * self.block_size - self._removed_before[wk_index]
                + local - self._removed_before_local(info, local))

    def __getitem__(self, index):
        flat = np.ravel_multi_index(index, self.shape) if isinstance(index, tuple) else int(index)
        return self._flat[self.flat_index(flat)]


def convert_dense_table(filename, base_name, shape, piece_names, chunk_size=1_048_576):
    """Write and fully validate compact temporary files; leave the source intact."""
    shape = tuple(shape)
    full_count = int(np.prod(shape, dtype=np.int64))
    block_size = int(np.prod(shape[1:], dtype=np.int64))
    if shape[0] != 10 or block_size != 64**4 * 2:
        raise ValueError(f"Unexpected non-pawn five-man shape for {base_name}: {shape}")
    if chunk_size % 8:
        raise ValueError("chunk_size must be divisible by 8")
    dtype = np.dtype(np.int8 if os.path.getsize(filename) == full_count else np.int16)
    if os.path.getsize(filename) != full_count * dtype.itemsize:
        raise ValueError(f"Unexpected dense table size for {base_name}")

    same_12 = piece_names[0] == piece_names[1]
    same_23 = piece_names[1] == piece_names[2]
    # Match _canonical_flat_5's three-pass bubble sort, including its final
    # (0, 1) pass when all three pieces have the same type.
    same_pairs = ([(0, 1)] if same_12 else []) + ([(1, 2)] if same_23 else []) + ([(0, 1)] if same_12 else [])
    diagonal_wk_indices = (0, 2, 5, 9)
    source = np.memmap(filename, dtype=dtype, mode="r", shape=(full_count,))
    compact_tmp = filename + ".compact.tmp"
    map_tmp = filename + ".diagmap.tmp"
    block_metadata = []
    payload_parts = []
    compact_count = 0

    try:
        with open(compact_tmp, "wb") as output:
            for wk_index in range(shape[0]):
                block_start = wk_index * block_size
                if wk_index not in diagonal_wk_indices:
                    source[block_start:block_start + block_size].tofile(output)
                    compact_count += block_size
                    continue

                mask = np.zeros(block_size // 8, dtype=np.uint8)
                kept_values = []
                removed_count = 0
                for start in range(0, block_size, chunk_size):
                    end = min(start + chunk_size, block_size)
                    local = np.arange(start, end, dtype=np.int64)
                    p1, p2, p3, bk = _decode_local(local)
                    keep = _transpose_canonical_keep(
                        np.full(local.size, wk_index, dtype=np.int8),
                        [p1, p2, p3, bk], same_pairs,
                    )
                    # The generator's retained side has tuple <= transpose tuple.
                    removed = ~keep
                    if removed.any():
                        mask[start // 8:(end // 8)] = np.packbits(removed, bitorder="little")
                        aliases = np.flatnonzero(removed).astype(np.int64) + start
                        a1, a2, a3, abk = _decode_local(aliases)
                        t1, t2, t3, tbk = (_transpose(a1), _transpose(a2), _transpose(a3), _transpose(abk))
                        for left, right in same_pairs:
                            vals = [t1, t2, t3, tbk]
                            swap = vals[left] > vals[right]
                            vals[left], vals[right] = np.where(swap, vals[right], vals[left]), np.where(swap, vals[left], vals[right])
                            t1, t2, t3, tbk = vals
                        turns = aliases & 1
                        alias_local = _encode_local(t1, t2, t3, tbk, turns)
                        old_vals = source[block_start + aliases]
                        alias_vals = source[block_start + alias_local]
                        # Dense legacy files may already leave redundant mirror
                        # cells zero. Any stored nonzero mirror value must agree.
                        mismatch = (old_vals != 0) & (old_vals != alias_vals)
                        if mismatch.any():
                            bad = int(np.flatnonzero(mismatch)[0])
                            raise ValueError(
                                f"Diagonal symmetry mismatch in {base_name}, flat "
                                f"{block_start + int(aliases[bad])}"
                            )
                    kept_values.append(source[block_start + start:block_start + end][keep].copy())
                    removed_count += int(removed.sum())
                for values in kept_values:
                    values.tofile(output)
                compact_count += block_size - removed_count

                # Prefix counts allow rank lookup with at most 4095 mask bits scanned.
                bits_per_rank = RANK_BLOCK_BITS
                mask_2d = mask.reshape(-1, bits_per_rank // 8)
                removed_per_rank = _BIT_COUNTS[mask_2d].sum(axis=1, dtype=np.uint64)
                prefix = np.empty(removed_per_rank.size + 1, dtype="<u4")
                prefix[0] = 0
                np.cumsum(removed_per_rank, dtype=np.uint64, out=prefix[1:])
                block_metadata.append({
                    "wk_index": wk_index,
                    "removed_count": removed_count,
                    "prefix": prefix,
                    "mask": mask,
                })

        # Build the compact map's JSON header and binary payload.
        payload = bytearray()
        blocks = []
        for info in block_metadata:
            prefix_bytes = info["prefix"].tobytes()
            prefix_offset = len(payload)
            payload.extend(prefix_bytes)
            mask_offset = len(payload)
            payload.extend(info["mask"].tobytes())
            blocks.append({
                "wk_index": info["wk_index"],
                "removed_count": info["removed_count"],
                "prefix_offset": prefix_offset,
                "prefix_count": int(info["prefix"].size),
                "mask_offset": mask_offset,
            })
        meta = {
            "format": 1,
            "shape": list(shape),
            "dtype": dtype.str,
            "full_count": full_count,
            "compact_count": compact_count,
            "block_size": block_size,
            "payload_offset": HEADER.size,
            "blocks": blocks,
        }
        # Payload offset depends on the JSON header length. Rebuild to a stable value.
        for _ in range(4):
            header_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
            meta["payload_offset"] = HEADER.size + len(header_bytes)
        header_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        with open(map_tmp, "wb") as handle:
            handle.write(HEADER.pack(MAGIC, len(header_bytes)))
            handle.write(header_bytes)
            handle.write(payload)

        expected_bytes = compact_count * dtype.itemsize
        if os.path.getsize(compact_tmp) != expected_bytes:
            raise ValueError(f"Compact byte-count mismatch for {base_name}")
        # Compare the packed stream to the exact selected source values in chunks.
        packed = np.memmap(compact_tmp, dtype=dtype, mode="r", shape=(compact_count,))
        packed_cursor = 0
        for wk_index in range(shape[0]):
            block_start = wk_index * block_size
            if wk_index in diagonal_wk_indices:
                for start in range(0, block_size, chunk_size):
                    end = min(start + chunk_size, block_size)
                    local = np.arange(start, end, dtype=np.int64)
                    p1, p2, p3, bk = _decode_local(local)
                    keep = _transpose_canonical_keep(np.full(local.size, wk_index), [p1, p2, p3, bk], same_pairs)
                    expected = source[block_start + start:block_start + end][keep]
                    next_cursor = packed_cursor + expected.size
                    if not np.array_equal(expected, packed[packed_cursor:next_cursor]):
                        raise ValueError(f"Compact round-trip mismatch in {base_name}")
                    packed_cursor = next_cursor
            else:
                next_cursor = packed_cursor + block_size
                if not np.array_equal(source[block_start:block_start + block_size], packed[packed_cursor:next_cursor]):
                    raise ValueError(f"Uncompressed block mismatch in {base_name}, king index {wk_index}")
                packed_cursor = next_cursor
        packed._mmap.close()
        if packed_cursor != compact_count:
            raise ValueError(f"Packed position count mismatch for {base_name}")

        # Exercise the persisted rank map against the still-available dense source.
        probe = DiagonalCompressedTable.__new__(DiagonalCompressedTable)
        probe.filename = compact_tmp
        probe.map_filename = map_tmp
        probe.shape = shape
        probe.ndim = len(shape)
        probe.base_name = base_name
        with open(map_tmp, "rb") as handle:
            _, header_size = HEADER.unpack(handle.read(HEADER.size))
            probe.meta = json.loads(handle.read(header_size).decode("utf-8"))
        probe.dtype = dtype
        probe.full_count = full_count
        probe.compact_count = compact_count
        probe.block_size = block_size
        probe._flat = np.memmap(compact_tmp, dtype=dtype, mode="r", shape=(compact_count,))
        probe._map = np.memmap(map_tmp, dtype=np.uint8, mode="r")
        probe._blocks = {int(item["wk_index"]): item for item in probe.meta["blocks"]}
        probe._removed_before = {}
        removed_before = 0
        for wk_index in range(shape[0]):
            probe._removed_before[wk_index] = removed_before
            if wk_index in probe._blocks:
                removed_before += int(probe._blocks[wk_index]["removed_count"])
        rng = np.random.default_rng(0)
        for wk_index in range(shape[0]):
            local = rng.integers(0, block_size, size=2048, dtype=np.int64)
            if wk_index in diagonal_wk_indices:
                p1, p2, p3, bk = _decode_local(local)
                keep = _transpose_canonical_keep(np.full(local.size, wk_index), [p1, p2, p3, bk], same_pairs)
                local = local[keep]
            flats = wk_index * block_size + local
            mapped = np.fromiter((probe.flat_index(int(flat)) for flat in flats), dtype=np.int64, count=flats.size)
            if not np.array_equal(source[flats], probe._flat[mapped]):
                raise ValueError(f"Rank-map lookup mismatch in {base_name}, king index {wk_index}")
        probe._flat._mmap.close()
        probe._map._mmap.close()
        return compact_tmp, map_tmp, compact_count, dtype.itemsize
    except Exception:
        for temporary in (compact_tmp, map_tmp):
            if os.path.exists(temporary):
                os.remove(temporary)
        raise
    finally:
        source._mmap.close()


def expand_compressed_table(filename, output_filename, shape, base_name, chunk_size=1_048_576):
    """Expand a compact table, filling removed diagonal aliases with zeroes."""
    table = DiagonalCompressedTable(filename, shape, base_name)
    full = np.memmap(output_filename, dtype=table.dtype, mode="w+", shape=(table.full_count,))
    cursor = 0
    try:
        for wk_index in range(shape[0]):
            block_start = wk_index * table.block_size
            info = table._blocks.get(wk_index)
            if info is None:
                end = cursor + table.block_size
                full[block_start:block_start + table.block_size] = table._flat[cursor:end]
                cursor = end
                continue
            mask_offset = table.meta["payload_offset"] + int(info["mask_offset"])
            for start in range(0, table.block_size, chunk_size):
                end = min(start + chunk_size, table.block_size)
                first_byte, last_byte = start // 8, end // 8
                removed = np.unpackbits(
                    table._map[mask_offset + first_byte:mask_offset + last_byte],
                    bitorder="little",
                ).astype(bool, copy=False)
                keep_local = np.flatnonzero(~removed) + start
                next_cursor = cursor + keep_local.size
                full[block_start + keep_local] = table._flat[cursor:next_cursor]
                cursor = next_cursor
        if cursor != table.compact_count:
            raise ValueError(f"Expanded position count mismatch for {base_name}")
        full.flush()
    except Exception:
        full._mmap.close()
        if os.path.exists(output_filename):
            os.remove(output_filename)
        raise
    finally:
        if getattr(full, "_mmap", None) is not None:
            full._mmap.close()
        table._flat._mmap.close()
        table._map._mmap.close()
    return table.dtype
