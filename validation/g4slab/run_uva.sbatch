#!/bin/bash
# R3 (approved 2026-11-21): Geant4 slab references (hepsimlib-recorder, FTFP_BERT_EMZ) and the hep-simkernels
# comparison. CPU only. usage:
#   sbatch -A <account> -c <cores> --mem=<mem> -t <time> -o <log> run_uva.sbatch WORK SIMKERNELS SIMLIB G4_SIF ANA_SIF SCALE
# WORK gets build/ and run/; SCALE 0.02 is the pilot, 1 the campaign. Geant4 shards run one per core.
set -euo pipefail
W=$(readlink -f ${1:?work}); K=$(readlink -f ${2:?hep-simkernels}); L=$(readlink -f ${3:?hep-simlib})
G4=${4:?geant4 image}; ANA=${5:?analysis image}; SCALE=${6:?scale}
NC=${SLURM_CPUS_PER_TASK:-1}
module load apptainer
mkdir -p $W/run
echo "hep-simkernels $(git -C $K log -1 --format=%h), hep-simlib $(git -C $L log -1 --format=%h), cores $NC"; date
if [ ! -x $W/build/hepsimlib-recorder ]; then
  apptainer exec --cleanenv -B $W -B $L $G4 bash -c "cmake -S $L/recorder -B $W/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/conda > $W/build.log 2>&1 && cmake --build $W/build -j $NC >> $W/build.log 2>&1" \
    || { tail -40 $W/build.log; exit 1; }
fi
apptainer exec --cleanenv -B $W -B $K $ANA python3 $K/validation/g4slab/campaign.py $W/run --scale $SCALE
export W G4
ls $W/run/*.mac | xargs -P $NC -I{} bash -c 'b=$(basename {} .mac); [ -s $W/run/$b.root ] && exit 0;
  /usr/bin/time -f "TIME $b %e s %M kB" apptainer exec --cleanenv -B $W --pwd $W/run $G4 $W/build/hepsimlib-recorder \
  FTFP_BERT_EMZ $b.mac > $W/run/$b.log 2> $W/run/$b.time || { echo FAIL $b; tail -5 $W/run/$b.log; exit 255; }'
cat $W/run/*.time | grep TIME | sort -k3 -n | tail -5
date
E="apptainer exec --cleanenv -B $W -B $K -B $L $ANA env PYTHONPATH=$K:$L PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu"
$E python3 $K/validation/g4slab/analyze.py $W/run --arms null,numpy --output $W/rec_np.json &
$E env HEPSIMK_DEVICE_FLOAT=float64 python3 $K/validation/g4slab/analyze.py $W/run --arms device --output $W/rec_d64.json &
$E env HEPSIMK_DEVICE_FLOAT=float32 python3 $K/validation/g4slab/analyze.py $W/run --arms device --output $W/rec_d32.json &
wait
$E python3 $K/validation/g4slab/analyze.py --combine $W/rec_np.json $W/rec_d64.json $W/rec_d32.json --output $W/record.json
date
