#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_KEY_GENERAL_FILE_TYPE = "general.file_type"

# Legacy Prism enum values before GGML_TYPE_Q1_0_g128 was inserted.
LEGACY_GGML_TYPE_MAP = {
    41: 42,  # old GGML_TYPE_Q1_0 -> current GGML_TYPE_Q1_0
    42: 43,  # old GGML_TYPE_Q2_0 -> current GGML_TYPE_Q2_0
}

LEGACY_GGML_FTYPE_MAP = {
    27: 28,  # old GGML_FTYPE_MOSTLY_Q1_0 -> current GGML_FTYPE_MOSTLY_Q1_0
    28: 29,  # old GGML_FTYPE_MOSTLY_Q2_0 -> current GGML_FTYPE_MOSTLY_Q2_0
}

TYPE_NAMES = {
    0: "F32",
    1: "F16",
    34: "TQ1_0",
    35: "TQ2_0",
    41: "legacy-Q1_0 / current-Q1_0_g128",
    42: "legacy-Q2_0 / current-Q1_0",
    43: "current-Q2_0",
}

FTYPE_NAMES = {
    27: "legacy-MOSTLY_Q1_0 / current-MOSTLY_Q1_0_g128",
    28: "legacy-MOSTLY_Q2_0 / current-MOSTLY_Q1_0",
    29: "current-MOSTLY_Q2_0",
}

SCALAR_SIZES = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


@dataclass
class TensorTypePatch:
    index: int
    name: str
    offset: int
    old_type: int
    new_type: int


@dataclass
class FileTypePatch:
    offset: int
    scalar_type: int
    old_value: int
    new_value: int


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, got {len(data)}")
    return data


def read_u32(handle) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_i32(handle) -> int:
    return struct.unpack("<i", read_exact(handle, 4))[0]


def read_u64(handle) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_i64(handle) -> int:
    return struct.unpack("<q", read_exact(handle, 8))[0]


def read_string(handle) -> str:
    length = read_u64(handle)
    return read_exact(handle, length).decode("utf-8")


def skip_scalar_value(handle, scalar_type: int, count: int = 1) -> None:
    if scalar_type == GGUF_TYPE_STRING:
        for _ in range(count):
            _ = read_string(handle)
        return

    size = SCALAR_SIZES.get(scalar_type)
    if size is None:
        raise ValueError(f"unsupported GGUF scalar type {scalar_type}")
    handle.seek(size * count, 1)


def parse_gguf(path: Path) -> tuple[FileTypePatch | None, list[TensorTypePatch]]:
    tensor_patches: list[TensorTypePatch] = []
    file_type_patch: FileTypePatch | None = None

    with path.open("rb") as handle:
        if read_exact(handle, 4) != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")

        version = read_u32(handle)
        if version != GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version {version}, expected {GGUF_VERSION}")

        n_tensors = read_u64(handle)
        n_kv = read_u64(handle)

        for _ in range(n_kv):
            key = read_string(handle)
            value_type = read_i32(handle)

            if value_type == GGUF_TYPE_ARRAY:
                element_type = read_i32(handle)
                count = read_u64(handle)
                skip_scalar_value(handle, element_type, count)
                continue

            value_offset = handle.tell()

            if key == GGUF_KEY_GENERAL_FILE_TYPE and value_type in (4, 5):
                old_value = read_u32(handle) if value_type == 4 else read_i32(handle)
                new_value = LEGACY_GGML_FTYPE_MAP.get(old_value)
                if new_value is not None:
                    file_type_patch = FileTypePatch(value_offset, value_type, old_value, new_value)
            else:
                skip_scalar_value(handle, value_type)

        for index in range(n_tensors):
            name = read_string(handle)
            n_dims = read_u32(handle)
            for _ in range(n_dims):
                _ = read_i64(handle)

            type_offset = handle.tell()
            old_type = read_i32(handle)
            _ = read_u64(handle)  # tensor data offset within data blob

            new_type = LEGACY_GGML_TYPE_MAP.get(old_type)
            if new_type is not None:
                tensor_patches.append(TensorTypePatch(index, name, type_offset, old_type, new_type))

    return file_type_patch, tensor_patches


def apply_patches(destination: Path, file_type_patch: FileTypePatch | None, tensor_patches: list[TensorTypePatch]) -> None:
    with destination.open("r+b") as handle:
        if file_type_patch is not None:
            handle.seek(file_type_patch.offset)
            if file_type_patch.scalar_type == 4:
                handle.write(struct.pack("<I", file_type_patch.new_value))
            else:
                handle.write(struct.pack("<i", file_type_patch.new_value))

        for patch in tensor_patches:
            handle.seek(patch.offset)
            handle.write(struct.pack("<i", patch.new_type))


def default_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.fixed{path.suffix}")


def describe_type(type_id: int) -> str:
    return TYPE_NAMES.get(type_id, str(type_id))


def describe_ftype(ftype_id: int) -> str:
    return FTYPE_NAMES.get(ftype_id, str(ftype_id))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite legacy Prism GGUF Q1_0/Q2_0 tensor type IDs to the current ggml enum. "
            "This is intended for files produced before GGML_TYPE_Q1_0_g128 was inserted."
        )
    )
    parser.add_argument("input", type=Path, help="Input GGUF file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output GGUF file. Defaults to <input>.fixed.gguf unless --in-place is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Patch the input file directly instead of writing a copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the changes that would be made without writing anything.",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=8,
        help="How many tensor patches to print in the summary (default: 8).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.in_place and args.output is not None:
        raise SystemExit("--in-place and --output are mutually exclusive")

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"input file not found: {input_path}")

    file_type_patch, tensor_patches = parse_gguf(input_path)

    if file_type_patch is None and not tensor_patches:
        print(f"no legacy Q1_0/Q2_0 ids found in {input_path}")
        return 0

    print(f"input: {input_path}")
    if file_type_patch is not None:
        print(
            "general.file_type: "
            f"{file_type_patch.old_value} ({describe_ftype(file_type_patch.old_value)}) -> "
            f"{file_type_patch.new_value} ({describe_ftype(file_type_patch.new_value)})"
        )

    print(f"tensor type patches: {len(tensor_patches)}")
    for patch in tensor_patches[: max(0, args.show_limit)]:
        print(
            f"  tensor[{patch.index}] {patch.name}: "
            f"{patch.old_type} ({describe_type(patch.old_type)}) -> "
            f"{patch.new_type} ({describe_type(patch.new_type)})"
        )
    if len(tensor_patches) > args.show_limit:
        print(f"  ... {len(tensor_patches) - args.show_limit} more")

    if args.dry_run:
        print("dry-run: no changes written")
        return 0

    destination = input_path if args.in_place else (args.output.resolve() if args.output else default_output_path(input_path))
    if destination != input_path:
        shutil.copyfile(input_path, destination)

    apply_patches(destination, file_type_patch, tensor_patches)
    print(f"wrote fixed GGUF: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())