
nsteps=2000000

for T in {080..110..002}  ; do

  BASE="T$T"

  if ! [ -d $BASE ] ; then
      mkdir $BASE
  fi 


  for i in {001..999} ; do

    DIR="$BASE/$i"

#   cp MD/run_sim.sh $DIR
#   cp MD/md.mdp $DIR
#   mv $DIR/runs/traj.xtc $DIR/runs/traj_01.xtc
#   mkdir $DIR/runs
#   cp $DIR/traj.xtc $DIR/runs/traj_01.xtc

#   cp MD/gromacs.gro $DIR

    echo $DIR
    echo "$DIR" >> directory.list

    if ! [ -d $DIR ] ; then
        cp -r MD $DIR
    else
        cp -r MD/* $DIR
    fi 



    sed "6i nsteps = $nsteps" $DIR/md.mdp | sed "7d"  > tmp; mv tmp $DIR/md.mdp

    sed "44i ref_t = $T" $DIR/md.mdp | sed "45d"  > tmp; mv tmp $DIR/md.mdp
    sed "49i gen_temp = $T" $DIR/md.mdp | sed "50d"  > tmp; mv tmp $DIR/md.mdp

  done
done

