import os 
import glob 
import numpy 
import pandas 


# =====================================================================
#     Create Workspace
# =====================================================================
def create_directory_comparision(main_path : str, flames : list)-> None : 
    
    if os.path.exists(os.path.join(main_path,"Detailed")) == False : 
        os.makedirs(os.path.join(main_path,"Detailed"))
        
       
    if os.path.exists(os.path.join(main_path,"Reduced")) == False : 
        os.makedirs(os.path.join(main_path,"Reduced"))
        
    if os.path.exists(os.path.join(main_path,"Error")) == False : 
        os.makedirs(os.path.join(main_path,"Error"))
    for f in flames : 
        if os.path.exists(os.path.join(main_path,f"Error/{f}")) == False :
            os.makedirs(os.path.join(main_path,f"Error/{f}"))
            os.makedirs(os.path.join(main_path,f"Error/{f}/Scaler"))
            os.makedirs(os.path.join(main_path,f"Error/{f}/Data"))
            os.makedirs(os.path.join(main_path,f"Error/{f}/Plot"))
            os.makedirs(os.path.join(main_path,f"Error/{f}/Err"))
    
    