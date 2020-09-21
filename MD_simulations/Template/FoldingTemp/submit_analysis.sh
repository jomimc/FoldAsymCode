#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -v ENSH_HOME=/engrid/ensh
# $ -q cslm.q@molandn01
# $ -pe ompi_ef 28
#$ -N foldAnal
#$ -t 1-15984

BASE=$(pwd)
DIR=$(sed -n "${SGE_TASK_ID}p" $BASE/directory.list)

date
pwd
source /home/jmcbride/VEnv3/bin/activate


  cd $BASE/$DIR/

  EXE="/home/jmcbride/CotransFold/Src/analysis.py"
  python $EXE -q 

date

