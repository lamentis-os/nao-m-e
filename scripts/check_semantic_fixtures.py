#!/usr/bin/env python3
"""Run and compare the fixed-profile semantic CI fixtures."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


MAGIC = b"nao-m-e-e5-fixture-v1\0"
PROFILE_FINGERPRINT = bytes.fromhex(
    "295a32ca1455fd8d81bfd42f4b950b2c0e037b6ab4717656db25d8300c1a6d4e"
)
DIMENSIONS = 384
HEADER_SIZE = len(MAGIC) + 32 + struct.calcsize("<IH")
TEST_NAME = "pinned_runtime_smoke_is_explicit_and_repeatable"


def read_fixture(path: Path) -> tuple[bytes, int, tuple[int, ...], bytes]:
    data = path.read_bytes()
    if len(data) < HEADER_SIZE or data[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path}: invalid semantic fixture header")

    offset = len(MAGIC)
    fingerprint = data[offset : offset + 32]
    if fingerprint != PROFILE_FINGERPRINT:
        raise ValueError(f"{path}: unexpected semantic profile fingerprint")
    offset += 32
    count, dimensions = struct.unpack_from("<IH", data, offset)
    offset += struct.calcsize("<IH")
    if count == 0:
        raise ValueError(f"{path}: fixture contains no vectors")
    if dimensions != DIMENSIONS:
        raise ValueError(
            f"{path}: expected {DIMENSIONS} dimensions, found {dimensions}"
        )

    component_count = count * dimensions
    expected_size = offset + component_count * struct.calcsize("<h")
    if len(data) != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} bytes, found {len(data)}"
        )
    values = struct.unpack_from(f"<{component_count}h", data, offset)
    return fingerprint, count, values, data


def run_fixture(output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    first = output.with_name(f"{output.name}.first")
    second = output.with_name(f"{output.name}.second")
    for path in (output, first, second):
        path.unlink(missing_ok=True)

    command = [
        "cargo",
        "test",
        "-p",
        "nao-m-e-semantic",
        "--release",
        "--locked",
        "--test",
        "runtime",
        TEST_NAME,
        "--",
        "--ignored",
        "--exact",
    ]
    for path in (first, second):
        environment = os.environ.copy()
        environment["NAO_M_E_SEMANTIC_FIXTURE_PATH"] = str(path)
        subprocess.run(command, check=True, env=environment)
        read_fixture(path)

    if first.read_bytes() != second.read_bytes():
        raise RuntimeError("separate semantic runtime processes produced different bytes")
    shutil.move(first, output)
    second.unlink()
    print(
        f"same-platform-byte-equality=true "
        f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}"
    )


def cosine(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("fixture contains a zero vector")
    return dot / (left_norm * right_norm)


def compare_fixtures(directory: Path) -> None:
    paths = sorted(directory.rglob("*.bin"))
    if len(paths) != 3:
        raise ValueError(f"expected 3 platform fixtures, found {len(paths)}")

    fixtures = {path.stem: read_fixture(path) for path in paths}
    fingerprints = {fixture[0] for fixture in fixtures.values()}
    counts = {fixture[1] for fixture in fixtures.values()}
    if len(fingerprints) != 1:
        raise ValueError("platform fixtures use different profile fingerprints")
    if len(counts) != 1:
        raise ValueError("platform fixtures contain different vector counts")
    count = counts.pop()

    for name, (_, _, _, data) in fixtures.items():
        print(
            f"fixture={name} vectors={count} dimensions={DIMENSIONS} "
            f"bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}"
        )

    all_bytes_equal = True
    violations: list[str] = []
    for (left_name, left), (right_name, right) in itertools.combinations(
        fixtures.items(), 2
    ):
        left_values, right_values = left[2], right[2]
        max_delta = max(abs(a - b) for a, b in zip(left_values, right_values))
        minimum_cosine = 1.0
        for index in range(count):
            start = index * DIMENSIONS
            end = start + DIMENSIONS
            minimum_cosine = min(
                minimum_cosine,
                cosine(left_values[start:end], right_values[start:end]),
            )
        byte_equal = left[3] == right[3]
        all_bytes_equal &= byte_equal
        print(
            f"pair={left_name},{right_name} byte_equal={str(byte_equal).lower()} "
            f"max_component_delta={max_delta} min_cosine={minimum_cosine:.12f}"
        )
        if max_delta > 1:
            violations.append(
                f"{left_name}/{right_name}: component delta {max_delta} exceeds 1"
            )
        if minimum_cosine < 0.999999:
            violations.append(
                f"{left_name}/{right_name}: cosine {minimum_cosine:.12f} "
                "is below 0.999999"
            )

    print(f"cross-platform-byte-equality={str(all_bytes_equal).lower()}")
    if violations:
        raise RuntimeError("; ".join(violations))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "run":
        run_fixture(arguments.output)
    else:
        compare_fixtures(arguments.directory)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"semantic fixture check failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
