
### First remove all HETATM and other records manually
### Some HETATM records are actaully AAs, in which case
### convert non-standard AAs to standard AAs


python ../../../ChaperoneProtein/gokit-master/conmaps.py  --gconmap 1ilo.pdb

python ../../../ChaperoneProtein/gokit-master/gokit.py -attype 1 --aa_pdb  1ilo.pdb -skip_glycine

cp contacts.txt external_pair

cd MD

  cp gromacs.top  ../../FoldingTemp/MD

  cp gromacs.gro reference.gro
  cp gromacs.top reference.top

  python /home/jmcbride/CotransFold/Init/protein_plus_wall.py --mode init --gro reference.gro --top reference.top  -i 1

  cp * ../../N_first_01/MD
  cp * ../../C_first_01/MD

  vi gromacs.gro
  cp gromacs.gro  ../../FoldingTemp/MD

cd ../

