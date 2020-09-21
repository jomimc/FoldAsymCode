#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -v ENSH_HOME=/engrid/ensh
#$ -q cslm.q
# $ -pe ompi_ef 28
#$ -N fold_3bid
#$ -t 1-15984
# $ -hold_jid FoldTemp14

BASE=$(pwd)
DIR=$(sed -n "${SGE_TASK_ID}p" $BASE/directory.list)

date
pwd
source /home/jmcbride/VEnv3/bin/activate

echo $DIR

 cd $BASE/$DIR/
 rm ./#* traj* state*
 ./run_gro.sh

  EXE="/home/jmcbride/CotransFold/Src/analysis.py"
  python $EXE -q 

date

