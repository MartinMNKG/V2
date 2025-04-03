from main_mpi import optim_prob
import os 
import shutil 

pop_size = 10
ngen = 100 
mutpb = 0.3
cxpb  = 1 
elitism_size = int(pop_size*10/100)
fitness = "PyOptMECH" # Dummy  or PyOptMECH

file_detailed = "detailed.yaml"
file_reduced = "reduced.yaml"

mode= "Start" # Restart Start

optim_prob(pop_size,ngen,mutpb,cxpb,elitism_size,file_detailed,file_reduced,fitness,mode)


