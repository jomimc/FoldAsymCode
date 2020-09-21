#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -v ENSH_HOME=/engrid/ensh
# $ -q cslm.q@molandn01
# $ -pe ompi_ef 28
#$ -N AnalFun
#$ -t 1-23976
# $ -t 1311-1441


BASE=$(pwd)
DIR=$(sed -n "${SGE_TASK_ID}p" $BASE/directory.list)

#EXE="/home/jmcbride/ChaperoneProtein/gokit-master/conmaps.py"
EXE="/home/jmcbride/CotransFold/Src/analysis.py"

date
pwd
echo $DIR

source /home/jmcbride/VEnv3/bin/activate

PDB="/home/jmcbride/CotransFold/3GN5/PDB/3gn5.pdb"
CA="/home/jmcbride/CotransFold/3GN5/PDB/native_ca.pdb"
PAIR="/home/jmcbride/CotransFold/3GN5/PDB/external_pair"


# cd $BASE/$DIR/

  python $EXE -q  -d $BASE/$DIR

# python /home/jmcbride/CotransFold/Init/write_topology.py gromacs.gro
# python $EXE --aa_pdb $PDB --native $CA --traj_list traj.xtc --cutoff 4.5 --scaling 1.2 --ext_pair $PAIR  --frames 1,100000,1
# python /home/jmcbride/ChaperoneProtein/gokit-master/conmaps.py --pl_fe traj.xtc.map

date

