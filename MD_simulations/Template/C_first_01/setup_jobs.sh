T=100

#rm anal.list
 rm directory.list

for steps in $(cat steps.txt)  ; do


  BASE="trans_time_$steps"
  if ! [ -d $BASE ] ; then
      mkdir $BASE
  fi 

  for i in {0001..4000} ; do

    DIR="$BASE/$i"

#   cp MD/run_sim.sh $DIR

#   for i in {001..131} ;  do 
#     echo "$DIR/RES$i" >> anal.list
#   done

    echo $DIR
    echo "$DIR" >> directory.list

    if ! [ -d $DIR ] ; then
        cp -r MD $DIR
    else
        cp -r MD/* $DIR
    fi 



    sed "6i nsteps = $steps" $DIR/md.mdp | sed "7d"  > tmp; mv tmp $DIR/md.mdp

    sed "44i ref_t = $T" $DIR/md.mdp | sed "45d"  > tmp; mv tmp $DIR/md.mdp
    sed "49i gen_temp = $T" $DIR/md.mdp | sed "50d"  > tmp; mv tmp $DIR/md.mdp

  done
done

