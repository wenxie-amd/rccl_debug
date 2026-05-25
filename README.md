# rccl_debug

A small RCCL allreduce determinism checker.

`run.sh` invokes `rccl_debug.py` back-to-back several times (default: 10):

1. **First iteration** (`--first-run 1`): rank 0 generates fresh bf16 input
   tensors and writes them to disk; all participating ranks load their
   tensor and run allreduce; rank 0 saves the result as the reference.
2. **All later iterations** (`--first-run 0`): every rank reloads the
   exact same inputs from disk, runs allreduce again, and rank 0 compares
   the output against the saved reference. Because both the inputs and the
   RCCL communicator setup are identical, the result must match bit-for-bit
   on every iteration. Any mismatch means RCCL is non-deterministic and is
   reported as `MISMATCH`.

Assumes the runtime container is already up and you are inside it, in this
directory.

## Single node (8 GPUs)

```bash
./run.sh
```

Optional arguments (forwarded to every iteration):

```bash
./run.sh --ranks 0,3,4,7        # only some ranks participate (rank 0 required)
./run.sh --tensor-size 200MB    # per-rank tensor size (used on iteration 1)
./run.sh --tensor-dir /tmp/out  # where input_*.pt + reference_output.pt live
ITERATIONS=20 ./run.sh          # change the number of iterations (default 10)
```

## Two nodes (16 GPUs)

Launch on each node, pointing `MASTER_ADDR` at the rank-0 node:

Node 0 (`mi355-gpu-8`):
```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=mi355-gpu-8 MASTER_PORT=29500 ./run.sh
```

Node 1 (`mi355-gpu-26`):
```bash
NNODES=2 NODE_RANK=1 MASTER_ADDR=mi355-gpu-8 MASTER_PORT=29500 ./run.sh
```

Or, under Slurm:
```bash
srun --ntasks-per-node=1 ./run.sh
```

For multi-node runs, `--tensor-dir` must be a path visible on every node
(default `./output`, which lives under the shared `/shared` mount).

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--ranks` | all ranks | Comma-separated participating ranks; rank 0 must be included |
| `--tensor-size` | `100MB` | Per-rank tensor size (`KB` / `MB` / `GB` accepted) |
| `--tensor-dir` | `./output` | Directory for input tensors (must be shared across nodes) |

## Output

Per iteration, rank 0 prints one of:

- `[rank 0] saved reference allreduce output to ...` — iteration 1 only.
- `[rank 0] PASS: allreduce output matches first-run reference exactly ...`
  — this iteration's allreduce produced the exact same bits as iteration 1.
- `[rank 0] MISMATCH: allreduce output differs from first-run reference --
  RCCL is non-deterministic for identical inputs` — followed by mismatch
  stats (count, max/mean abs diff, first 20 differing positions). This is
  a **real** RCCL bug, not bf16 rounding noise, because both runs used the
  exact same inputs.

`run.sh` aborts on the first iteration whose torchrun returns non-zero
(NCCL error, file-not-found, etc.). Otherwise it prints
`all N iterations completed` at the end. Per-node output is also written to
`logs/rankN.log`.

## Environment overrides

Every NCCL / RCCL variable in `run.sh` uses `${VAR:-default}`, so callers
can override individually:

```bash
NCCL_DEBUG=INFO ./run.sh                # verbose NCCL logs
NCCL_IB_HCA=rocep9s0:1 ./run.sh         # restrict to a single IB NIC
MASTER_PORT=29600 ./run.sh              # change the rendezvous port
```

See `run.sh` for the full list of defaults.
