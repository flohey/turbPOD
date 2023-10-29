import numpy as np
import torch
import time
import yaml
import os
import sys

sys.path.append(os.getcwd()+"/../../")
from pod_class import POD, _DTYPE

def main(config):
    time_start = time.time()

    #-------------
    # 1. Config
    #-------------
    pod = POD(config)
    print(f"Computing POD of {pod.nfields} fields. Data dimensions: nx={pod.nx}, ny={pod.ny}, nz={pod.nz}, nt={pod.nt}. Aspect ratio: {pod.aspect}.")   
    
    #---------------------
    #2. Import Data
    #---------------------
    print('Importing data & adapting to requirements.')
    t_import1 = time.time()
    data = torch.from_numpy(np.load(pod.datapath)).to(_DTYPE)    # read from .npy file
    data = data.reshape(pod.nt,pod.nfields,pod.nz,pod.ny,pod.nx)  # reshape to fit POD class requirements
    t_import2 = time.time()
    print('Time taken: {:.2f} min'.format((t_import2 - t_import1)/60))
    
    #--------
    #3. POD
    #--------
    print('Computing POD')
    t_svd1 = time.time()
    time_coefficients, spatial_modes, eigen_values, mean = pod.apply(data,save=False)
    t_svd2 = time.time()
    print('Time taken: {:.2f} min'.format((t_svd2 - t_svd1)/60))

    #---------------
    #4. Save Results
    #---------------
    print('Saving')
    t_save1 = time.time()
    pod.save(time_coefficients, spatial_modes, eigen_values, mean)
    t_save2 = time.time()
    print(f"POD saved to {pod.savefile}")   
    print('Time taken: {:.2f} min'.format((t_save2 -t_save1)/60))
    

    time_end = time.time()
    print('\n ----------------------------------------')
    print('\n PROGRAM FINISHED!')
    print('\n ----------------------------------------')
    print('\n Total elapsed time {0:.2f}s'.format(time_end-time_start))
    


if __name__ == '__main__':

    # Load YAML config file
    #---------------------------
    with open("pod_config.yaml","r") as f:
        yaml_config = yaml.safe_load(f)
        
    main(yaml_config)



#---------------------------------------------------------------------------------------------
#                          END OF PROGRAM
#---------------------------------------------------------------------------------------------
