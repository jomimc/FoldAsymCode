#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -v ENSH_HOME=/engrid/ensh
#$ -q cslm.q
# $ -pe ompi_ef 28
#$ -N N03
#$ -t 1-28000

BASE=$(pwd)
DIR=$(sed -n "${SGE_TASK_ID}p" $BASE/directory.list)

date
pwd
source /home/jmcbride/VEnv3/bin/activate
#python $BASE/Src/generate_native_config.py

echo $DIR

cd $BASE/$DIR/


INIT="/home/jmcbride/CotransFold/Init"
n_steps=1500000

i=131

  sed "6i nsteps = $n_steps" md.mdp | sed "7d"  > tmp; mv tmp md.mdp

  ./run_gro.sh

  SAVE="RES$(printf %03d $i)"
  cp  gromacs.gro  gromacs.top  confout.gro  traj.xtc  md.log  traj.trr  $SAVE


  rm ./#* traj* state*

  EXE="/home/jmcbride/CotransFold/Src/anal_probe.py"
  python $EXE -q  -d $BASE/$DIR

date


