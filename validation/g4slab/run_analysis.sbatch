#!/bin/bash
# R3 campaign, analysis stage: one array task per (case, arm); arms: numpy (with the null), device float64, float32.
#   sbatch -A <account> -p standard -c 1 --mem 8G -t 01:30:00 --array=0-29 -o <log> run_analysis.sbatch WORK SIMKERNELS SIMLIB ANA_SIF
# then: analyze.py --combine WORK/rec_*.json --output WORK/record.json
# Measured (campaign, 1M Geant4 events per case, 200k device): numpy arm 20 min-1:01, device 1:36-8:30 except
# float64 in W (0.35 mm 1:14, 1 X0 0:47); MaxRSS 2.2-5.2 GB -> ask 6G and 1:30.
set -euo pipefail
W=$(readlink -f ${1:?work}); K=$(readlink -f ${2:?hep-simkernels}); L=$(readlink -f ${3:?hep-simlib}); ANA=${4:?analysis image}
module load apptainer
cases=($(apptainer exec --cleanenv -B $W $ANA python3 -c "import json; print(' '.join(c['case'] for c in json.load(open('$W/run/campaign.json'))['cases']))"))
i=${SLURM_ARRAY_TASK_ID:?array}; c=${cases[$((i / 3))]}; arm=$((i % 3))
E="apptainer exec --cleanenv -B $W -B $K -B $L $ANA env PYTHONPATH=$K:$L PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu"
case $arm in
  0) $E python3 $K/validation/g4slab/analyze.py $W/run --cases $c --arms null,numpy --output $W/rec_${c}_np.json ;;
  1) $E env HEPSIMK_DEVICE_FLOAT=float64 python3 $K/validation/g4slab/analyze.py $W/run --cases $c --arms device --output $W/rec_${c}_d64.json ;;
  2) $E env HEPSIMK_DEVICE_FLOAT=float32 python3 $K/validation/g4slab/analyze.py $W/run --cases $c --arms device --output $W/rec_${c}_d32.json ;;
esac
