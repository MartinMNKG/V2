import os 
import glob 
import sys 
import time
from utils.Database import  charger_parametres
from utils.path import create_directory_comparision
from utils.flame import launch_speedflame_pure,launch_0Dreactor_pure,launch_counterflow_pure
from utils.flame import launch_speedflame_bifuel,launch_0Dreactor_bifuel,launch_counterflow_bifuel
from utils.error import launch_change_grid,launch_error_comparision
from utils.plot import plot_comparision

if __name__ =="__main__" : 
    
    # =====================================================================
    #     INPUT/OUTPUT INITIALIZATION
    # =====================================================================
    start_time = time.time()
    main_path = os.getcwd()
    file_ref = "/work/kotlarcm/WORK/Tools_Database/Tools1_Comparision_0D/Skel/input/Detailed.yaml" # Detailed File 
    file_red = "/work/kotlarcm/WORK/Tools_Database/Tools1_Comparision_0D/Skel/input/Reduced.yaml"# Reduced File 
    
    file_input = "/work/kotlarcm/WORK/Tools_Database/Tools1_Comparision_0D/Skel/input/input_database.json" # Input Database (SF / 0D / CF )
    main_path = os.getcwd()
    
    info = "Comparision" #Or Classification 
    
    flames = ["0Dreactor"] #(SF / 0D / CF )
    
    # =====================================================================
    #     MAIN / SECOND FUEL 
    # =====================================================================
    
    # Main fuel 
    fuel ="NH3"
         
    #Second fuel if Mixture 
    bifuel = True
    second_fuel = "H2"
    
    # =====================================================================
    #     SPECIES / INFO CANTERA (SF / 0D / CF )
    # =====================================================================

    species_of_interest = ["H2", "NH3","H2O", "OH","N2O","NO","NO2"] #["CH4","O2","CO2","H2O","CO"]
    
    # Commun Grid (SF / 0D / CF )
    lenght = 100 
    
    #For 0D reactor : 
    dt = 1e-6
    end_time = 0.3
    mode ="equi"
    
    #Launch Data base 
    verbose = True 
    
   
    
    create_directory_comparision(main_path,flames)
    for f in flames : 
        
        if bifuel == False : 
            temp,pressure,second_param = charger_parametres(file_input,f)
        if bifuel == True :
            temp,pressure,second_param, mixture = charger_parametres(file_input,f) 
            print(mixture) 
                       
        if f == "speedflame" : 
            
            print("#"*10)
            print(f"{f}")
            print(temp,pressure,second_param)
            if verbose == True : 
                if bifuel == False : 
                    launch_speedflame_pure(temp,pressure,second_param,file_ref,file_red,fuel,main_path)
                if bifuel == True  :
                    launch_speedflame_bifuel(temp,pressure,second_param,mixture,file_ref,file_red,fuel,second_fuel,main_path)
            
            # Launch change grid 
            launch_change_grid(main_path,lenght,f,info,bifuel,fuel)
            # Launch calcul Error
            launch_error_comparision(main_path,species_of_interest,f,bifuel)
            # Launch species Error 
            plot_comparision(main_path,species_of_interest,f)
            #Launch IDT Error
        
        elif f == "0Dreactor" :
            print("#"*10)
            print(f"{f}")
            print(temp,pressure,second_param)
            if bifuel == True: 
                print(mixture) 
                           
            if verbose == True :
                if bifuel == False : 
                    launch_0Dreactor_pure(temp,pressure,second_param,file_ref,file_red,fuel,main_path,dt,end_time,mode)
                elif bifuel == True :
                    launch_0Dreactor_bifuel(temp,pressure,second_param,mixture,file_ref,file_red,fuel,second_fuel,main_path,dt,end_time,mode)
           
            # Launch change grid 
            launch_change_grid(main_path,lenght,f,info,bifuel,fuel)
            # Launch calcul Error
            launch_error_comparision(main_path,species_of_interest,f,bifuel)
            # Launch species Error 
            plot_comparision(main_path,species_of_interest,f)
            

        elif f =="counterflow"  : 
            print("#"*10)
            print(f"{f}")
            print(temp,pressure,second_param)
            
            if verbose == True :
                if bifuel == True : 
                    launch_counterflow_bifuel(temp,pressure,second_param,mixture,file_ref,file_red,fuel,second_fuel,main_path)
            
                else : 
                    launch_counterflow_pure(temp,pressure,second_param,file_ref,file_red,fuel,main_path)
            
            # # Launch change grid 
            # launch_change_grid(main_path,lenght,f,info,bifuel,fuel)
            # # Launch calcul Error
            # launch_error_comparision(main_path,species_of_interest,f,bifuel)
            # # Launch species Error 
            # plot_comparision(main_path,species_of_interest,f)
           
    print("#"*10)
    print(f"Calculs time :{time.time()-start_time} ")
    
    
    
    