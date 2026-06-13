# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequentialize sflow's slurm backend allocation (local workaround).

Why this exists
---------------
``sflow.app.assembly.allocate_backends`` allocates every backend concurrently
(``asyncio.gather``), but ``SlurmBackend.allocate()`` parses ``salloc`` output
through a shared module-level logger. With two or more slurm backends, the
concurrent allocations' ``salloc: Nodes <nodelist> are ready`` lines flow
through *every* backend's parser, so ``parsed_result["nodelist"]`` becomes a
list instead of a string and ``scontrol getaddrs <list>`` crashes with::

    TypeError: expected string or bytes-like object, got 'list'

This patches ``allocate_backends`` to allocate backends one at a time, so each
backend's parser only sees its own ``salloc`` output. Both allocations are
still held concurrently afterward; only the ``salloc`` calls are serialized.

qwen36_smg_sflow.yaml needs this (two slurm backends: gpu_cluster + cpu_cluster).
Single-backend workflows never hit the bug and don't need the patch.

Usage
-----
Run with the SAME Python that runs sflow (activate that venv first)::

    python examples/09_MultiTurn/patch_sflow_seqalloc.py            # apply
    python examples/09_MultiTurn/patch_sflow_seqalloc.py --revert   # undo
    python examples/09_MultiTurn/patch_sflow_seqalloc.py --check    # report only

Idempotent: re-running is safe (no-op once applied). Writes a ``.bak`` next to
the patched file. Re-apply after any sflow reinstall/upgrade. This is an
upstream bug — remove the workaround once sflow fixes concurrent multi-backend
allocation.
"""

import argparse
import importlib.util
import pathlib
import sys

MARKER = "PATCHED: sequential slurm allocation"

OLD = (
    "    tasks = [b.allocate_resources() for b in to_allocate]\n"
    "    results = await asyncio.gather(*tasks, return_exceptions=True)\n"
)

NEW = (
    "    # PATCHED: sequential slurm allocation. Concurrent salloc output is parsed\n"
    "    # through a shared module logger, so >1 concurrent allocation cross-\n"
    "    # contaminates the nodelist parse (list instead of str) and scontrol\n"
    "    # getaddrs crashes. Serialize the salloc calls so each parser sees only\n"
    "    # its own output. Both allocations are still held concurrently afterward.\n"
    "    results = []\n"
    "    for _b in to_allocate:\n"
    "        try:\n"
    "            await _b.allocate_resources()\n"
    "            results.append(None)\n"
    "        except Exception as _e:  # noqa: BLE001\n"
    "            results.append(_e)\n"
)


def _assembly_path() -> pathlib.Path:
    """Locate the installed sflow's app/assembly.py via the active interpreter."""
    spec = importlib.util.find_spec("sflow")
    if spec is None or spec.submodule_search_locations is None:
        sys.exit(
            "sflow is not importable in this Python — activate the venv that runs sflow."
        )
    locs = list(spec.submodule_search_locations)
    if not locs:
        sys.exit("could not resolve the sflow package location")
    return pathlib.Path(locs[0]) / "app" / "assembly.py"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sequentialize sflow slurm backend allocation "
        "(workaround for the concurrent-salloc nodelist crash)."
    )
    ap.add_argument("--revert", action="store_true", help="restore the .bak backup")
    ap.add_argument("--check", action="store_true", help="report patch status only")
    args = ap.parse_args()

    f = _assembly_path()
    src = f.read_text()
    bak = f.with_suffix(".py.bak")
    patched = MARKER in src

    if args.check:
        print(f"{f}\nstatus: {'PATCHED' if patched else 'unpatched'}")
        return 0

    if args.revert:
        if not patched:
            print(f"not patched, nothing to revert: {f}")
            return 0
        if not bak.exists():
            sys.exit(f"no backup found at {bak} — revert manually")
        f.write_text(bak.read_text())
        bak.unlink()
        print(f"reverted: {f}")
        return 0

    if patched:
        print(f"already patched: {f}")
        return 0
    if OLD not in src:
        sys.exit(
            f"expected code block not found in {f}\n"
            "sflow may have changed — patch allocate_backends by hand."
        )
    bak.write_text(src)
    f.write_text(src.replace(OLD, NEW, 1))
    print(f"patched: {f}\nbackup:  {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
