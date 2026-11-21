#!/bin/bash
# R2: hep-simkernels vs ESTAR and PDG muon tables, both device floats. NumPy and CPU JAX only.
# Measured (2026-11-21): 30-38 s elapsed, MaxRSS 1.4-1.6 GB on one core; requested 3G / 5 min.
#SBATCH -J simk-estar-pdg
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=3G
#SBATCH -t 00:05:00
# usage: sbatch -A <account> -o <log> run_uva.sbatch <hep-simkernels checkout> <hep-simlib checkout> <image.sif>
set -euo pipefail
K=${1:?hep-simkernels}; L=${2:?hep-simlib}; SIF=${3:?image}
module load apptainer
cd "$K"
for dt in float64 float32; do
  apptainer exec "$SIF" env PYTHONPATH="$K:$L" HEPSIMK_DEVICE_FLOAT=$dt JAX_PLATFORMS=cpu \
    python3 validation/estar_pdg/compare.py --out "validation/estar_pdg/record_$dt.json"
done
apptainer exec "$SIF" env PYTHONPATH="$K:$L" python3 validation/estar_pdg/compare.py \
  --report validation/estar_pdg/record_float64.json validation/estar_pdg/record_float32.json
