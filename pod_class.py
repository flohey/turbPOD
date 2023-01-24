import numpy as np
import torch
import h5py
import time
import os


_DTYPE = torch.float32

class POD:
    
    def __init__(self,config=None,savefile=None):
        """
        INPUT:
            config   - yaml config file
            savefile - hdf5 file containing previously computed POD
        """
        assert (config is not None or savefile is not None), "Error: config and savefile can't both be None"
        
        if config is not None:
            self.init_from_config(config)
        else:
            self.init_from_file(savefile)
            
        self.nfields=len(self.fieldnames)
        self.total_points = int(self.nx*self.ny*self.nz)
        
        if self.nz > 1:
            self.set_up_3d()
        else:
            self.set_up_2d()
        
    def init_from_config(self,config):
        
            self.savefile = os.path.join(config["savepath"],config["savename"])
            self.datapath = config["datapath"]

            self.nt = config["nt"]
            self.aspect = config["aspect"]
            self.nx = config["nx"]
            self.ny = config["ny"]
            self.nz = config.setdefault('nz', 1)
            self.fieldnames = tuple(config["fieldnames"])
            self.use_lateral_fft = config["use_lateral_fft"]
            self.use_svd = config["use_svd"]
            self.subtract_diffusive_profile = list(config.setdefault("subtract_diffusive_profile",list()))
    
    def init_from_file(self,savefile):
            
            self.savefile = savefile
            self.datapath = None
            with h5py.File(self.savefile,'r') as f:
                self.nt = f.attrs["nt"]
                self.aspect = f.attrs["aspect"]
                self.nx = f.attrs["nx"]
                self.ny = f.attrs["ny"]
                self.nz = f.attrs["nz"]
                G = f['fieldnames']
                self.fieldnames = tuple(G.keys())
                
                self.use_lateral_fft = bool(f.attrs["use_lateral_fft"])
                self.use_svd = bool(f.attrs["use_svd"])
                self.subtract_diffusive_profile = list(f["subtract_diffusive_profile"].keys())
    
    def set_up_3d(self):
        
            self.fft = torch.fft.fft2
            self.ifft = torch.fft.ifft2
            self.fft_dim = (3,4)
            self.lta_dim = (0,2,3)
            
            xTab = torch.linspace(-self.aspect/2,self.aspect/2,self.nx)
            yTab = torch.linspace(-self.aspect/2,self.aspect/2,self.ny)
            self.vertical_coord = torch.linspace(0,1,self.nz)
            Z,_,_  = torch.meshgrid(self.vertical_coord,yTab,xTab,indexing='ij')
            self.diffusive_profile = 1 - Z.reshape(self.nz,self.ny,self.nx)
    
    def set_up_2d(self):
            self.fft = torch.fft.fft
            self.ifft = torch.fft.ifft
            self.fft_dim = 4
            self.lta_dim = (0,1,3)
            
            xTab = torch.linspace(-self.aspect/2,self.aspect/2,self.nx)
            self.vertical_coord = torch.linspace(0,1,self.ny)
            Y,_  = torch.meshgrid(self.vertical_coord,xTab,indexing='ij')
            self.diffusive_profile = 1-Y.reshape(self.nz,self.ny,self.nx)
            
    #----------------------------------------------------------
    def read_nek_data(self,datapath=None):
        """
        Read interpolated data from Nek5000 run.
        """
        
        if datapath is None:
            assert self.datapath is not None, "Error no datapath was provided."
            datapath = self.datapath
            
        x = torch.empty((self.nt,self.nfields,self.nz,self.ny,self.nx))
        for ifield, name in enumerate(self.fieldnames):
            for ii,it in enumerate(range(1,self.nt+1)):
                x[ii,ifield]  = torch.tensor(np.fromfile(datapath+name+'.dat_{:05d}'.format(it)).reshape(self.nz,self.ny,self.nx)).to(_DTYPE)
        
        return x

    #----------------------------------------------------------
    def apply(self,data=None,save=False):
        """
        Perform POD on data & save to self.savefile.
        """
        
        #Subtract diffusive profiles
        #---------------------------
        assert data.shape == (self.nt,self.nfields,self.nz,self.ny,self.nx), "Error: data has an incorrect shape. Please provide (nt,nfields,nz,ny,nx): " +f"(({self.nt},{self.nfields},{self.nz},{self.ny},{self.nx}))"
        for ifield, name in enumerate(self.fieldnames):
            if name == self.subtract_diffusive_profile:
                data[:,ifield] = data[:,ifield] - self.diffusive_profile.unsqueeze(0)
        
        # Subtract time mean <>_t
        #-------------------------
        mean = data.mean(dim=(0,)).reshape(1,self.nfields,self.nz,self.ny,self.nx)
        data = data - mean

        # Perform FFT in lateral directions
        #---------------------------------
        if self.use_lateral_fft:
            data = self.fft(data,dim = self.fft_dim)

        data = data.reshape(self.nt,self.nfields*self.total_points)

        if self.use_svd:
            time_coefficients, singular_values, _ = torch.linalg.svd(1/torch.sqrt(self.nt-1)*data)  
            eigen_values = singular_values**2
        else:
            eigen_values, time_coefficients = torch.linalg.eigh(1/(self.nt-1)*data@data.t().conj())

        #Sort modes according to energy & thermal variance
        eigen_values, idx_sort = torch.sort(eigen_values,descending=True)
        time_coefficients = time_coefficients[:,idx_sort]
        
        # Project data on eigenspace 
        spatial_modes = torch.matmul(data.t().conj(),time_coefficients)

        if save:
            self.save(time_coefficients,spatial_modes,eigen_values,mean)
        
        return time_coefficients, spatial_modes, eigen_values, mean

    #----------------------------------------------------------
    def save(self,
              time_coefficients,
              spatial_modes,
              eigen_values,
              mean,
              compression='gzip',
              compression_opts=9):
        
        """
        Saves POD modes, time coefficients, singular values & time mean of physical fields.
        
        time_coefficients - 
        spatial_modes     -
        eigen_values      - eigenvalues of the covariance matrix
        mean              - time mean of physical fields <.>_t
        compression       - hdf5 compression algorithm
        compression_opts  - hdf5 compression level
        """
        
        with h5py.File(self.savefile, 'a') as f:
            f.create_dataset('mean', data=mean, compression=compression, compression_opts=compression_opts)
            f.create_dataset('eigen_values', data=eigen_values, compression=compression, compression_opts=compression_opts)
            f.create_dataset('time_coefficients', data=time_coefficients, compression=compression, compression_opts=compression_opts)
            f.create_dataset('spatial_modes', data=spatial_modes, compression=compression, compression_opts=compression_opts)
            
            G = f.create_group('subtract_diffusive_profile')
            for name in self.subtract_diffusive_profile:
                G.create_group(name)
            
            G = f.create_group('fieldnames')
            for name in self.fieldnames:
                G.create_group(name)
            
            f.attrs['use_lateral_fft'] = self.use_lateral_fft
            f.attrs['use_svd'] = self.use_svd
            f.attrs['nx'] = self.nx
            f.attrs['ny'] = self.ny
            f.attrs['nz'] = self.nz
            f.attrs['nt'] = self.nt
            f.attrs['aspect']  = self.aspect
            f.attrs['nfields'] = self.nfields

    #----------------------------------------------------------
    def get_spatial_modes(self,nmodes=None):
        """
        Returns spatial modes.
        INPUT:
            nmodes - no. spatial modes (if None, use all available)
        RETURN 
            spatial_modes - POD spatial modes
        """
        with h5py.File(self.savefile, "r") as f:    
            if nmodes is not None:
                spatial_modes = np.array(f["spatial_modes"][:,:nmodes])
            else:
                spatial_modes = np.array(f["spatial_modes"])
            
        return torch.from_numpy(spatial_modes).to(_DTYPE)
    
    #----------------------------------------------------------
    def get_mean(self):
        """
        Returns time mean <.>_t.
        RETURN 
            mean - time mean
        """
        with h5py.File(self.savefile, "r") as f:    
            mean = np.array(f["mean"])
            
        return torch.from_numpy(mean).to(_DTYPE)

    #----------------------------------------------------------
    def get_time_coefficients(self,nmodes=None,it_start=0,it_end=None):
        """
        Returns time coefficients.
        INPUT:
            nmodes - no. modes (if None, use all available)
        RETURN 
            time_coefficients - POD time coefficients
        """
        if it_end is None:
            it_end = self.nt
            
        with h5py.File(self.savefile, "r") as f:    
            if nmodes is None:
                _, nmodes = f["time_coefficients"].shape
            time_coefficients = np.array(f["time_coefficients"][it_start:it_end,:nmodes])
            
        return torch.from_numpy(time_coefficients).to(_DTYPE)

    #----------------------------------------------------------
    def reconstruct(self,x,spatial_modes=None,mean=None):
        """
        Reconstructs the physical fields from the POD time coefficients x and spatial modes (either loaded from pod_filepath or taken from arg)
        INPUT:
            x                 - time coefficients. Shape: (nt, nmodes)
            spatial_modes     - spatial modes, if None: read from hdf5 file
            mean              - time mean fields, if None: read from hdf5 file
        RETURN:
            reconstructed_fields - physical fields, reconstructed from POD time coefficients & spatial modes, shape (nt,nfields,nz,ny,nx)
        """

        nt, nmodes = x.shape

        if mean is None:
            with h5py.File(self.savefile, "r") as f:
                mean = np.array(f["mean"])
                mean = torch.from_numpy(mean).to(_DTYPE)

        if spatial_modes is None:
            with h5py.File(self.savefile, "r") as f:            
                spatial_modes = np.array(f["spatial_modes"][:,:nmodes])
                spatial_modes = torch.from_numpy(spatial_modes).to(_DTYPE)

        #Reconstruct from POD
        #---------------------
        reconstructed_data = x@spatial_modes[:,:nmodes].t().conj()                                # output shape (nt,nfields*nz*ny*nx)

        if self.use_lateral_fft:
            reconstructed_data = self.ifft(reconstructed_data.reshape(nt,self.nfields,self.nz,self.ny,self.nx),dim=self.fft_dim).real  # reverse FFT pre-processing, output shape (nt,nfields,nz,ny,nx)
        else:
            reconstructed_data = reconstructed_data.reshape(nt,self.nfields,self.nz,self.ny,self.nx).real # output shape (nt,nfields,nz,ny,nx)
        reconstructed_data += mean.reshape(1,self.nfields,self.nz,self.ny,self.nx)                             # add time mean fields
        
        for ifield,name in enumerate(self.fieldnames):
            if name in self.subtract_diffusive_profile:
                reconstructed_data[:,ifield] += self.diffusive_profile.unsqueeze(0)                            # add diffusive profile for temperature field
                
        reconstructed_data = torch.permute(reconstructed_data,(1,0,2,3,4))                                     # output shape (nfields,nt,nz,ny,nx)

        return reconstructed_data
    
    #----------------------------------------------------------
    def compute_lta(self,x1,x2=None):
        """
        Returns lateral-time-average profile (<x1^2>A,t)^1/2 or <x1*x2>A,t for given fields x1,x2.
        INPUT:
            x1 - field 1
            x2 - field 2 (optional)
        RETURN:
            lta - lateral-time average <>_A,t
        """

        assert x1.shape[1:] == (self.nz,self.ny,self.nx), "Expected x1 to be of shape (nt, nz, ny, nx)"
        if x2 is not None:
            assert x2.shape[1:] == (self.nz,self.ny,self.nx), "Expected x2 to be of shape (nt, nz, ny, nx)"
        
        # Compute lateral time avgerage
        #-------------------------------
        if x2 is None:
            lta = torch.sqrt(torch.mean(x1**2, dim=self.lta_dim)) # Root-mean-square (<x1^2>A,t)^1/2
        else:
            lta = torch.mean(x1*x2, dim=self.lta_dim)            # Covar. <x1*x2>_A,t

        return lta
        
    #----------------------------------------------------------
    def compute_nare(self,lta_truth,lta_test):
        """
        Compute normalized average relative error (NARE) of true and inferred lateral-time average profile.
        Definition see e.g.: Heyder & Schumacher PRE 103, 053107 (2021) eqs. (29-30)
        INPUT:
            lta_truth - lateral-time average of ground truth
            lta_test  - lateral-time average of inferred field
        RETURN:
            nare - normalized average relative error of lta profile
        """

        norm = 2*max(lta_truth)
        nare =  1/norm*torch.trapz(abs(lta_truth - lta_test), x=self.vertical_coord)   

        return nare
    
    #----------------------------------------------------------
