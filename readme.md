This repository contains Julia code and notebooks used to reproduce the computations and figures for the paper _Nonlinear dynamics of random neural networks with second-order synaptic motifs_.
## Repository Layout

- `MotifNets/`: Julia project environment. It contains `Project.toml` and
  `Manifest.toml`.
- `Utils.jl`: shared utilities.
- `DMFT.jl`: dynamic mean-field theory solvers and helper routines.
- `ComputingScripts/`: scripts used to generate numerical result files.
- `*.ipynb`: notebooks used for figures.
- `images/`: exported figure files.
- `results/`: generated data files. Precomputed results are available upon request.

## Julia Environment

Initialize the Julia environment from the repository root:

```bash
julia --project=MotifNets -e 'using Pkg; Pkg.instantiate()'
```

The notebooks should also be run with the `MotifNets` environment active.

Some utilities use `RCall` and the R package `diptest`. If a script or notebook
uses the bimodality/dip-test helpers, install `diptest` in the R installation
used by `RCall`.

## Reproducing Results

The computation scripts assume that the script being run, `Utils.jl`, and
`DMFT.jl` are in the same directory. To reproduce a result, place or copy the
relevant script from `ComputingScripts/` into a run directory together with
`Utils.jl` and, when the script uses DMFT routines, `DMFT.jl`.

Several long-running scripts use `Distributed` and `SlurmClusterManager` for
cluster execution. Those scripts should be run on a Slurm-enabled system with
`SlurmClusterManager.jl` available in the active Julia environment, or adapted
to use local workers.

## License

This code is released under the MIT License.
