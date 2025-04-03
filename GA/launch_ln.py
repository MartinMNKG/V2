from main_ln_mpi import optim_prob
import os 
import shutil 

pop_size = 5
ngen = 100 # 100 or 100 - N de Start
mutpb = 0.3
cxpb  = 1 
elitism_size = 1 #int(pop_size*10/100)
fitness = "Optismoke" # Dummy  or PyOptMECH Optismoke

file_detailed = "detailed.yaml"
file_reduced = "reduced.yaml"

mode= "Start" # Restart Start

optim_prob(pop_size,ngen,mutpb,cxpb,elitism_size,file_detailed,file_reduced,fitness,mode)


