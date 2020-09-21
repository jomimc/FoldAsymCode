#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -v ENSH_HOME=/engrid/ensh
#$ -q public.q@molandn34
#$ -pe ompi_ef 28
#$ -N AnalAmalg
# $ -t 16200-23976
# $ -t 1311-1441
#$ -hold_jid C03a


BASE=$(pwd)

EXE="/home/jmcbride/CotransFold/Src/analalysis.py"

date
pwd

source /home/jmcbride/VEnv3/bin/activate

 python $EXE -a --base_time 0 --seqlen 90


date

