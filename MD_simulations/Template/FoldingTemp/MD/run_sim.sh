


rm ./#* traj* state*


#for i in {01..10} ; do
i=03

  echo Running... $i

  ./run_gro.sh

  cp  gromacs.gro  gromacs.top  runs 
  cp  traj.xtc  runs/traj_${i}.xtc


  rm ./#* traj* state*

#done
