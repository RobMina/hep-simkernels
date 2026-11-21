#!/bin/bash
# R3 campaign, Geant4 stage: build hepsimlib-recorder in the public hep-simlib image and run every shard, one per core.
#   sbatch -A <account> -p standard -c 8 --mem 8G -t 01:30:00 -o <log> run_g4.sbatch WORK SIMKERNELS SIMLIB G4_SIF ANA_SIF SCALE
# Pilot (2 x 10k events per case): 5-10 s per 10k-event shard (4 s start-up), W 3.5 mm 4.8 ms/event, 0.2 GB per process.
set -euo pipefail
W=$(readlink -f ${1:?work}); K=$(readlink -f ${2:?hep-simkernels}); L=$(readlink -f ${3:?hep-simlib})
G4=${4:?geant4 image}; ANA=${5:?analysis image}; SCALE=${6:?scale}; NC=${SLURM_CPUS_PER_TASK:-1}
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
du -sh $W/run; date
