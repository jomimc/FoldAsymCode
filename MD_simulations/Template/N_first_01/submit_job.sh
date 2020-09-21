#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -v ENSH_HOME=/engrid/ensh
#$ -q cslm.q
# $ -pe ompi_ef 28
#$ -N N01_3bid
#$ -t 1-44000
# $ -t 75001-100000
# $ -t 75001-120000

BASE=$(pwd)
DIR=$(sed -n "${SGE_TASK_ID}p" $BASE/directory.list)

date
pwd
source /home/jmcbride/VEnv3/bin/activate
#python $BASE/Src/generate_native_config.py

echo $DIR

cd $BASE/$DIR/
./run_sim.sh


  EXE="/home/jmcbride/CotransFold/Src/analysis.py"
  python $EXE -q  -d $BASE/$DIR -g confout.gro --seqlen 58


date

