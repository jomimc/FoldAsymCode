

INIT="/home/jmcbride/CotransFold/Init"
nsteps=500000


for i in {1..58} ; do

  echo Running... $i

  if [ $i == 1 ] ; then
    python $INIT/protein_plus_wall.py --mode update  --top reference.top --gro reference.gro -i $i --reverse
  elif [ $i == 58 ] ; then
    python $INIT/protein_plus_wall.py --mode update  --top reference.top --gro confout.gro -i $i --free
    cp reference.top gromacs.top
  else
    python $INIT/protein_plus_wall.py --mode update  --top reference.top --gro confout.gro -i $i --reverse
  fi

  if [ $i == 2 ] ; then
    sed "48i gen_vel = no" md.mdp | sed "49d"  > tmp; mv tmp md.mdp
  fi

  if [ $i == 58 ] ; then
    sed "6i nsteps = $nsteps" md.mdp | sed "7d"  > tmp; mv tmp md.mdp
  fi

  ./run_gro.sh

  SAVE="RES$(printf %03d $i)"
  mkdir $SAVE
  cp  confout.gro  traj.xtc  md.log  $SAVE


  rm ./#* traj* state*

done
