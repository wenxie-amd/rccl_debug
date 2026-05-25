#!/usr/bin/env python3
"""RCCL allreduce determinism test.

One invocation of this script performs a single allreduce iteration. The
launcher (run.sh) calls it multiple times back-to-back:

    --first-run 1   (first iteration)
        Rank 0 generates N fresh bf16 tensors and writes them to
        ``<tensor-dir>/input_*.pt``. All participating ranks load their
        tensor and run allreduce. Rank 0 then writes the allreduce result
        to ``<tensor-dir>/reference_output.pt``.

    --first-run 0   (every subsequent iteration)
        All participating ranks load the SAME inputs from the first run and
        run allreduce again. Rank 0 compares the new result against
        ``reference_output.pt``. Because the inputs and the RCCL
        communicator setup are identical, the result MUST match exactly --
        any mismatch indicates RCCL non-determinism (a real bug, not bf16
        rounding noise).

Launch via torchrun; see run.sh.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


# Multi-node runs read tensors over a shared filesystem (typically NFS), where
# rank 0's freshly-saved files can take a moment to become visible on other
# hosts even after dist.barrier() returns.
NFS_LOAD_RETRY_SECONDS = 30.0
NFS_LOAD_RETRY_INTERVAL = 0.5


def save_tensor(t: torch.Tensor, path: Path) -> None:
    """Save tensor and fsync so it is durable on the NFS server."""
    torch.save(t, path)
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def wait_for_file(path: Path, timeout_s: float, interval_s: float) -> bool:
    """Poll path.is_file() until it appears or timeout. Returns False on timeout."""
    deadline = time.monotonic() + timeout_s
    while not path.is_file():
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_s)
    return True


def parse_size(s: str) -> int:
    """Parse '100MB', '1.5GB', '1024' (raw bytes) into bytes."""
    s = s.strip().upper()
    for unit, mult in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if s.endswith(unit):
            return int(float(s[: -len(unit)]) * mult)
    return int(s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCCL allreduce determinism test")
    p.add_argument(
        "--first-run",
        type=int,
        choices=[0, 1],
        required=True,
        help="1 = generate fresh inputs and save the allreduce output as the "
        "reference; 0 = reuse existing inputs and compare against the saved "
        "reference (must match exactly)",
    )
    p.add_argument(
        "--ranks",
        type=str,
        default=None,
        help="comma-separated participating ranks, e.g. '0,3,4,7'. "
        "Default: all ranks. Rank 0 must be included.",
    )
    p.add_argument(
        "--tensor-dir",
        type=str,
        default="./output",
        help="shared directory for tensor files (default: ./output)",
    )
    p.add_argument(
        "--tensor-size",
        type=str,
        default="100MB",
        help="per-rank tensor size, e.g. '100MB' or '1GB' (default: 100MB). "
        "Ignored when --first-run=0 (inputs already exist on disk).",
    )
    return p.parse_args()


def resolve_ranks(ranks_arg: str | None, world_size: int) -> tuple[list[int], str | None]:
    """Return (sorted_unique_ranks, error_message_or_None)."""
    if ranks_arg is None:
        return list(range(world_size)), None
    try:
        ranks = sorted({int(x) for x in ranks_arg.split(",") if x.strip()})
    except ValueError as e:
        return [], f"cannot parse --ranks '{ranks_arg}': {e}"
    if not ranks:
        return [], "--ranks parsed to an empty list"
    if 0 not in ranks:
        return [], f"rank 0 must participate, got: {ranks}"
    if ranks[-1] >= world_size:
        return [], f"participating ranks {ranks} exceed world size {world_size}"
    return ranks, None


def report_mismatch(reference: torch.Tensor, got: torch.Tensor, max_print: int = 20) -> None:
    """Pretty-print mismatch info between the allreduce output and the reference."""
    mismatch = got != reference
    n_mis = int(mismatch.sum().item())
    n_total = reference.numel()
    pct = 100.0 * n_mis / n_total
    print(f"  mismatch count : {n_mis}/{n_total} ({pct:.4f}%)")

    diff = (got.float() - reference.float()).abs()
    print(f"  max abs diff   : {diff.max().item():.6e}")
    print(f"  mean abs diff  : {diff.mean().item():.6e}")

    idxs = torch.nonzero(mismatch, as_tuple=True)[0]
    n_show = min(max_print, idxs.numel())
    if n_show == 0:
        return
    print(f"  first {n_show} mismatching positions:")
    print(f"    {'idx':>12} {'got':>16} {'expected':>16} {'abs_diff':>14}")
    for i in range(n_show):
        p = int(idxs[i].item())
        g = float(got[p].item())
        e = float(reference[p].item())
        print(f"    {p:>12d} {g:>16.6f} {e:>16.6f} {abs(g - e):>14.3e}")


def main() -> int:
    args = parse_args()

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(
            f"[rank 0] torch distributed init complete "
            f"(world_size={world_size}, first_run={args.first_run})"
        )

    participating_ranks, err = resolve_ranks(args.ranks, world_size)
    if err:
        if rank == 0:
            print(f"[error] {err}", file=sys.stderr)
        dist.destroy_process_group()
        return 1
    n_participants = len(participating_ranks)

    if rank == 0:
        print(
            f"[rank 0] participating ranks: {participating_ranks} "
            f"(count: {n_participants})"
        )

    # Collective on the global PG; every rank must call it.
    pg = dist.new_group(ranks=participating_ranks)

    dtype = torch.bfloat16
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = parse_size(args.tensor_size) // elem_size
    tensor_dir = Path(args.tensor_dir).resolve()
    ref_output_path = tensor_dir / "reference_output.pt"

    my_tensor: torch.Tensor | None = None

    # ---- Phase 1: produce inputs (first run only) ----
    if args.first_run == 1 and rank == 0:
        tensor_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[rank 0] generating fresh inputs: dtype={dtype}, numel={numel}, "
            f"size~{numel * elem_size / 1024**2:.1f}MB, dir={tensor_dir}"
        )
        in_mem: list[torch.Tensor] = []
        for i in range(n_participants):
            t = torch.empty(numel, dtype=dtype, device=device).uniform_(-1.0, 1.0)
            save_tensor(t.cpu(), tensor_dir / f"input_{i}.pt")
            in_mem.append(t)
        my_tensor = in_mem[0]
        print(f"[rank 0] wrote {n_participants} input tensors")

    # On the first iteration, non-rank-0 ranks must wait for rank 0's writes.
    # On subsequent iterations the files are already on disk; the barrier is
    # unnecessary but cheap and keeps the ordering identical between runs.
    dist.barrier()
    torch.cuda.synchronize()

    # ---- Phase 2: load inputs + allreduce ----
    if rank in participating_ranks:
        if my_tensor is None:
            idx = participating_ranks.index(rank)
            tensor_path = tensor_dir / f"input_{idx}.pt"
            if not wait_for_file(
                tensor_path, NFS_LOAD_RETRY_SECONDS, NFS_LOAD_RETRY_INTERVAL
            ):
                print(
                    f"[rank {rank}] error: tensor file not found after "
                    f"{NFS_LOAD_RETRY_SECONDS:.0f}s: {tensor_path}",
                    file=sys.stderr,
                )
                # Abort the whole job; surviving ranks would hang on allreduce.
                dist.destroy_process_group()
                return 2
            loaded = torch.load(tensor_path, map_location=device)
            my_tensor = loaded.to(dtype) if loaded.dtype != dtype else loaded

        assert my_tensor is not None
        output_tensor = my_tensor.clone()
        dist.all_reduce(output_tensor, op=dist.ReduceOp.SUM, group=pg)

    dist.barrier()
    torch.cuda.synchronize()

    # ---- Phase 3: save reference (first run) or compare to it ----
    if rank == 0:
        if args.first_run == 1:
            save_tensor(output_tensor.cpu(), ref_output_path)
            print(f"[rank 0] saved reference allreduce output to {ref_output_path}")
        else:
            if not wait_for_file(
                ref_output_path, NFS_LOAD_RETRY_SECONDS, NFS_LOAD_RETRY_INTERVAL
            ):
                print(
                    f"[rank 0] error: reference output not found at "
                    f"{ref_output_path} (was --first-run 1 ever executed?)",
                    file=sys.stderr,
                )
                dist.destroy_process_group()
                return 2
            reference = torch.load(ref_output_path, map_location=device)
            if reference.dtype != dtype:
                reference = reference.to(dtype)
            if torch.equal(output_tensor, reference):
                print(
                    f"[rank 0] PASS: allreduce output matches first-run "
                    f"reference exactly ({reference.numel()} elements)"
                )
            else:
                # Same inputs + same RCCL communicator should give bit-exact
                # results. A diff here is real non-determinism, not bf16 noise.
                print(
                    "[rank 0] MISMATCH: allreduce output differs from "
                    "first-run reference -- RCCL is non-deterministic for "
                    "identical inputs"
                )
                report_mismatch(reference, output_tensor)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
