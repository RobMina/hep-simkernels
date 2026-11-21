#!/bin/bash
# R2: hep-simkernels vs ESTAR and PDG muon tables, both device floats. NumPy and CPU JAX only.
# Sized for a new job type: one core, under a minute of compute expected; 4G / 30 min requested.
#SBATCH -J simk-estar-pdg
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
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
