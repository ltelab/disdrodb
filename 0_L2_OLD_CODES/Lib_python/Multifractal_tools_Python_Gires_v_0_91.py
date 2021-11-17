# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:29:30 2013

@author: gires
"""

import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt


############################################################################################
############################################################################################
#
# Author : Auguste GIRES (2013)      (auguste.gires@leesu.enpc.fr)
#
# Toolbox for performing in Python: 
#       - spectral analysis
#       - fractal analysis
#       - multifractal analysis (Trace Moment analysis, Double Trace Moment for Universal mulitfractals)
#       - discrete numerical simulations (beta model and Universal Multifractals)
#
# If used, at least one of these papers should be cited: 
# - Gires, A., Tchiguirinskaia, I., Schertzer, D., and Lovejoy, S.: Development and analysis of a simple model 
#   to represent the zero rainfall in a universal multifractal framework, Nonlin. Processes Geophys., 20, 343-356, 
#   doi:10.5194/npg-20-343-2013, 2013.
# - Gires, A., Tchiguirinskaia, I., Schertzer, D., and Lovejoy, S.: Influence of the zero-rainfall on the assessment 
#   of the multifractal parameters. Advances in Water Resources, 2012. 45: p. 13-25.
# - Gires, A., Tchiguirinskaia, I., Schertzer, D., and Lovejoy, S.: Analyses multifractales et spatio-temporelles 
#   des precipitations du modele Meso-NH et des donnees radar. Hydrological Sciences Journal-Journal Des Sciences 
#   Hydrologiques, 2011. 56(3): p. 380-396
############################################################################################
############################################################################################


# This script allows to perform a fractal analysis. 
# The following options are available : 
#      - 1D or 2D analysis
#      - Sample analysis or average analysis on sets of samples
#      - Possibility to take into account up to three scaling regime
#      - Possibility to display or not the graphics

def fractal_dimension(data,data_file_name,dim,l_range,file_index,file_name,plot_index):

    # Inputs :
    # - data : data is a numpy matrix that contain the data to be analysed. 
    #          Its meaning depends on dim
    #      - dim=1 : 1D analysis are performed
    #                data is a 2D matrix, each column is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                if one column, then a sample analysis is performed)
    #      - dim=2 : 2D analysis are performed
    #                data is a 3D matrix, each layer (i.e. data[:,:,s]) is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                (if one layer, then a sample analysis is performed)
    # - data_file_name : - if data_file_name =='' then "data" is used for the analysis
    #                    - otherwise : data_file_name is list of .csv file name
    #                                  the 1D or 2D field contained in each file is considered as a independant sample
    #                                  (used for large files)                
    # - l_range is a list of list. Each element contains two elements (reso_min and reso_max)
    #                and a fractal dimension is evaluated between reso_min and reso_max
    #                if l_range=[] then a single scaling regime is considered and the all the available resolution are used
    #                the max length of l_range is 3; ie no more than three scaling regime can be studied
    # - file_index : the computation of the moments might be quite long, and therefore can be recorded in a file 
    #                that can be used later
    #                file_index=0 --> nb_ones is recorded
    #                file_index=1 --> nb_ones is retrieved from an existing file
    # - file_name : the name of the file recorded or retrieved
    #               WARNING the file should have a .npy extension !!
    # - plot_index : the number of the first figure opened for graph display 
    #
    # Outputs : 
    #  - D1 : the fractal dimension of the scaling regime associated with l_range[0]
    #  - D2 : the fractal dimension of the scaling regime associated with l_range[1] 
    #  - D3 : the fractal dimension of the scaling regime associated with l_range[2]
    # Note 1 : the fractal dimension of "unused" scaling regime are returned as "nan"  
    # Note 2 : a ration of 2 is used in the upscaling proces
    
    
    # Evaluating l_max (maximum resolution) and nb_s (number of samples)
    if dim ==1:
        if data_file_name=='':
            l_max=data.shape[0]
            if len(data.shape)==1:
                nb_s=1
            else :    
                nb_s=data.shape[1]
        else:
            nb_s=len(data_file_name)
            l_max=np.loadtxt(data_file_name[0]).shape[0]
    elif dim==2:
        if data_file_name=='':
            l_max=data.shape[0]
            if len(data.shape)==2:
                nb_s=1
            else:
                nb_s=data.shape[2]
        else:
            nb_s=len(data_file_name)
            l_max=np.loadtxt(data_file_name[0],delimiter=';').shape[0]
    else:
        print('Error in fractal_dimension, wrong dim')
  
    # TO BE DONE : write an error message of the size is not a power of 2
    n_max=sp.log(l_max)/sp.log(2)
    nb_ones=np.zeros((n_max+1))
    l_list=np.zeros((n_max+1))    
    for n in range(sp.int0(n_max)+1):
        l_list[n]=sp.power(2,n)
     

    # STEP 1 : Assessing nb_ones according to the data_type 
    
    if file_index == 0:  # nb_ones is computed
        if dim == 1:
            if data_file_name=='':           
                nb_ones[n_max]=sp.count_nonzero(data)/nb_s             
                for n_l in range(sp.int0(n_max-1),-1,-1):
                    nb_el=data.shape[0]
                    data=(data[0:nb_el:2,:]+data[1:nb_el+1:2,:])/2 #Upscaling of the field
                    nb_ones[n_l]=sp.count_nonzero(data)/nb_s
            else: 
                for s in range(0,nb_s):
                    data_s=np.loadtxt(data_file_name[s],delimiter=';')
                    nb_ones[n_max]=nb_ones[n_max]+sp.count_nonzero(data_s)/nb_s
                    for n_l in range(sp.int0(n_max-1),-1,-1):
                        nb_el=data_s.shape[0]                    
                        data_s=(data_s[0:nb_el:2]+data_s[1:nb_el+1:2])/2        
                        nb_ones[n_l]=nb_ones[n_l]+sp.count_nonzero(data_s)/nb_s
        elif dim == 2 :
            if data_file_name == '' :           
                nb_ones[n_max]=sp.count_nonzero(data)/nb_s             
                for n_l in range(sp.int0(n_max-1),-1,-1):
                    nb_el=data.shape[0]
                    if nb_s == 1 :
                        data=(data[0:nb_el:2,0:nb_el:2]+data[0:nb_el:2,1:nb_el+1:2]+data[1:nb_el+1:2,0:nb_el:2]+data[1:nb_el+1:2,1:nb_el+1:2])/4 #Upscaling of the field
                    else:
                        data=(data[0:nb_el:2,0:nb_el:2,:]+data[0:nb_el:2,1:nb_el+1:2,:]+data[1:nb_el+1:2,0:nb_el:2,:]+data[1:nb_el+1:2,1:nb_el+1:2,:])/4 #Upscaling of the field
                    nb_ones[n_l]=sp.count_nonzero(data)/nb_s
            else: 
                for s in range(0,nb_s):
                    data_s=np.loadtxt(data_file_name[s],delimiter=';')
                    nb_ones[n_max]=nb_ones[n_max]+sp.count_nonzero(data_s)/nb_s
                    for n_l in range(sp.int0(n_max-1),-1,-1):
                        nb_el=data_s.shape[0]                    
                        data_s=(data_s[0:nb_el:2,0:nb_el:2]+data_s[0:nb_el:2,1:nb_el+1:2]+data_s[1:nb_el+1:2,0:nb_el:2]+data_s[1:nb_el+1:2,1:nb_el+1:2])/4 #Upscaling of the field        
                        nb_ones[n_l]=nb_ones[n_l]+sp.count_nonzero(data_s)/nb_s
        np.save(file_name,nb_ones)
    elif file_index == 1 :    # nb_ones is simply loaded
        nb_ones=np.load(file_name)
    else: 
        print('Error in fractal_dimension_1D, file_index not equal to 0 or 1')

    
    # STEP 2 : Evaluation of the fractal dimension for the various scaling regime (defined in l_range)
    # l_range is a list of list each containing 2 elt (reso_min and reso_max)
    # a linear regression is performed for each scaling regime
    
    y=sp.log(nb_ones)
    x=sp.log(l_list)
    if plot_index>0:
        plt.figure(plot_index)
        p1, = plt.plot(x,y,ls='None',marker='x',ms=9,mew=2,mfc='k',mec='k')     
        plt.xlabel(r'$\log(\lambda)$',fontsize=20,color='k')
        plt.ylabel(r'$\log(N_\lambda)$',fontsize=20,color='k')
        plt.title('Fractal dimension',fontsize=20,color='k')  
        ax=plt.gca()
        for xtick in ax.get_xticklabels():
            plt.setp(xtick,fontsize=14)
        for ytick in ax.get_yticklabels():
            plt.setp(ytick,fontsize=14)    


    nb_scale_reg = len(l_range) #Evaluation of the number of scaling regime    
    
    if nb_scale_reg == 0:
        a=sp.polyfit(x,y,1)   
        reg_lin=sp.poly1d(a)    
        D1=a[0]
        r2=sp.corrcoef(x,y)[0,1]**2
        D2=np.nan
        D3=np.nan        
        if plot_index>0:
            p2, = plt.plot([x[0],x[-1]],[reg_lin(x[0]),reg_lin(x[-1])],lw=2,color='k')
            plt.legend([p2],[r'$\mathit{D}_F\ =\ $'+str(sp.floor(D1*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2*100)/100)],loc=2,frameon=False)
    elif nb_scale_reg == 1:
        # A single scaling regime
        i_l_min=np.where(l_list==l_range[0][0])[0][0]
        i_l_max=np.where(l_list==l_range[0][1])[0][0]
        a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
        reg_lin=sp.poly1d(a)        
        D1=a[0]
        D2=np.nan
        D3=np.nan
        if plot_index>0:
            r2=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            p2, = plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='k')
            plt.legend([p2],[r'$\mathit{D}_F\ =\ $'+str(sp.floor(D1*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2*100)/100)],loc=2,frameon=False)
    elif nb_scale_reg == 2:
        # For the 1st scaling regime
        i_l_min=np.where(l_list==l_range[0][0])[0][0]
        i_l_max=np.where(l_list==l_range[0][1])[0][0]
        a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
        reg_lin=sp.poly1d(a)   
        if plot_index>0:
            r2_1=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            p2, = plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='b')
        D1=a[0]
        # For the 2nd scaling regime        
        i_l_min=np.where(l_list==l_range[1][0])[0][0]
        i_l_max=np.where(l_list==l_range[1][1])[0][0]
        a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
        reg_lin=sp.poly1d(a) 
        D2=a[0]
        D3=np.nan
        if plot_index>0:
            r2_2=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            p3, = plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='r')
            plt.legend([p2,p3],[r'$\mathit{D}_F\ =\ $'+str(sp.floor(D1*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1*100)/100),r'$\mathit{D}_F\ =\ $'+str(sp.floor(D2*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_2*100)/100)],loc=2,frameon=False)         
    elif nb_scale_reg == 3:
        # For the 1st scaling regime        
        i_l_min=np.where(l_list==l_range[0][0])[0][0]
        i_l_max=np.where(l_list==l_range[0][1])[0][0]
        a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
        reg_lin=sp.poly1d(a)        
        r2=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
        if plot_index>0:
            r2_1=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            p2, = plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='b')
        D1=a[0]
        # For the 2nd scaling regime        
        i_l_min=np.where(l_list==l_range[1][0])[0][0]
        i_l_max=np.where(l_list==l_range[1][1])[0][0]
        a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
        reg_lin=sp.poly1d(a)        
        if plot_index>0:
            r2_2=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            p3, = plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='r')
        D2=a[0]
        # For the 3rd scaling regime        
        i_l_min=np.where(l_list==l_range[2][0])[0][0]
        i_l_max=np.where(l_list==l_range[2][1])[0][0]
        a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
        reg_lin=sp.poly1d(a)   
        D3=a[0]
        if plot_index>0:
            r2_3=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            p4, = plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='g')
            plt.legend([p2,p3,p4],[r'$\mathit{D}_F\ =\ $'+str(sp.floor(D1*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1*100)/100),r'$\mathit{D}_F\ =\ $'+str(sp.floor(D2*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_2*100)/100),r'$\mathit{D}_F\ =\ $'+str(sp.floor(D3*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_3*100)/100)],loc=2,frameon=False)
    else :
        print('Error in fractal_dimension_1D, l_range has wrong size')
        

    plt.savefig('evaluation_fractal_dimension.png')
        
    
    return D1,D2,D3    
    


############################################################################################
############################################################################################

            
# This script allows to perform a TM analysis. 
# The following options are available : 
#      - 1D or 2D analysis
#      - Sample analysis or average analysis on sets of samples
#      - Possibility to take into account up to three scaling regime
#      - Possibility to display or not the graphics

def TM(data,q_values,data_file_name,dim,l_range,file_index,file_name,plot_index):

    # Inputs :
    # - data : data is a numpy matrix that contain the data to be analysed. 
    #          Its meaning depends on dim
    #      - dim=1 : 1D analysis are performed
    #                data is a 2D matrix, each column is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                if one column, then a sample analysis is performed)
    #      - dim=2 : 2D analysis are performed
    #                data is a 3D matrix, each layer (i.e. data[:,:,s]) is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                (if one layer, then a sample analysis is performed)
    # - q_values : a numpy vector containing the different q for which K(q) is evaluated
    #          if q_values=np.array([-1]), then a default value is set np.concatenate((np.arange(0.05,1,0.05),np.array([1.01]),np.array([1.05]),np.arange(1.1,3.1,0.1)),axis=0)          
    #               (it is advised to use this option) 
    # - data_file_name : - if data_file_name =='' then "data" is used for the analysis
    #                    - otherwise : data_file_name is list of .csv file name
    #                                  the 1D or 2D field contained in each file is considered as a independant sample
    #                                  (used for large files)                
    # - l_range is a list of list. Each element contains two elements (reso_min and reso_max)
    #                and a fractal dimension is evaluated between reso_min and reso_max
    #                if l_range=[] then a single scaling regime is considered and the all the available resolution are used
    #                the max length of l_range is 3; ie no more than three scaling regime can be studied
    # - file_index : the computation of the moments might be quite long, and therefore can be recorded in a file 
    #                that can be used later
    #                file_index=0 --> "moments" is recorded
    #                file_index=1 --> "moments" is retrieved from an existing file
    # - file_name : the name of the file recorded or retrieved
    #               WARNING the file should have a .npy extension !!
    # - plot_index : the number of the first figure opened for graph display 
    #
    # Outputs : 
    #  - Kq_1 : the K(q) function of the scaling regime associated with l_range[0] for the values in q_values
    #  - Kq_2 : the K(q) function of the scaling regime associated with l_range[1] for the values in q_values
    #  - Kq_3 : the K(q) function of the scaling regime associated with l_range[2] for the values in q_values
    #  - r2_1 : the values of r2 for each q obtained in the linear regression leading to K(q) for the 1st scaling regime
    #  - r2_2 : the values of r2 for each q obtained in the linear regression leading to K(q) for the 2nd scaling regime
    #  - r2_3 : the values of r2 for each q obtained in the linear regression leading to K(q) for the 3rd scaling regime
    #
    # Note 1 : the output of "unused" scaling regime(s) are returned as "nan"  
    # Note 2 : a ration of 2 is used in the upscaling proces
    
    
    # Evaluating l_max (maximum resolution) and nb_s (number of samples)
    if dim ==1:
        if data_file_name=='':
            l_max=data.shape[0]
            if len(data.shape)==1:
                nb_s=1
            else :    
                nb_s=data.shape[1]
        else:
            nb_s=len(data_file_name)
            l_max=np.loadtxt(data_file_name[0]).shape[0]
    elif dim==2:
        if data_file_name=='':
            l_max=data.shape[0]
            if len(data.shape)==2:
                nb_s=1
            else:
                nb_s=data.shape[2]
        else:
            nb_s=len(data_file_name)
            l_max=np.loadtxt(data_file_name[0],delimiter=';').shape[0]
    else:
        print('Error in TM, wrong dim')
  
    # TO BE DONE : write an error message of the size is not a power of 2
    n_max=sp.log(l_max)/sp.log(2)
    l_list=np.zeros((n_max+1))    
    for n in range(sp.int0(n_max)+1):
        l_list[n]=sp.power(2,n)
     
    # Affecting the value of q_values if the default option has been selected
    # The initial value is stored in q_values_bis (which not is a numpy array for simplicity)
    if q_values.shape[0]==1 and q_values[0] == -1:
        q_values=np.concatenate((np.arange(0.05,1,0.05),np.array([1.01]),np.array([1.05]),np.arange(1.1,3.1,0.1)),axis=0)
        q_values_bis=-1
    else :
        q_values_bis=1
    
    nb_q = q_values.shape[0]
    Kq_1 = np.zeros((nb_q,))
    Kq_2 = np.zeros((nb_q,))
    Kq_3 = np.zeros((nb_q,))
    r2_1 = np.zeros((nb_q,))
    r2_2 = np.zeros((nb_q,))
    r2_3 = np.zeros((nb_q,))

    moments=np.zeros((n_max+1,nb_q))
    
    # Step 1 : Evaluate the moments <R_lambda^q>
    #          Results are stored in a numpy array moments 
    #                               (1st index --> resolution,)
    #                               (2nd index --> value of q, same order as in q_values )
    
    if file_index == 0:  # moments is computed

        if dim ==1:
            if data_file_name=='':
                for n_q in range(nb_q) :               
                    moments[n_max,n_q]=np.mean(np.power(data,q_values[n_q]))
                for n_l in range(sp.int0(n_max-1),-1,-1):
                    nb_el=data.shape[0]
                    data=(data[0:nb_el:2,:]+data[1:nb_el+1:2,:])/2 # Upscaling of the field
                    for n_q in range(nb_q) :               
                        moments[n_l,n_q]=np.mean(np.power(data,q_values[n_q]))
            else: 
                for s in range(nb_s):
                    data_s=np.loadtxt(data_file_name[s],delimiter=';') 
                    for n_q in range(nb_q) :               
                        moments[n_max,n_q]=moments[n_max,n_q]+np.mean(np.power(data_s,q_values[n_q]))/nb_s
                    for n_l in range(sp.int0(n_max-1),-1,-1):
                        nb_el=data_s.shape[0]
                        data_s=(data_s[0:nb_el:2]+data_s[1:nb_el+1:2])/2 # Upscaling of the field
                        for n_q in range(nb_q) :               
                            moments[n_l,n_q]=moments[n_l,n_q]+np.mean(np.power(data_s,q_values[n_q]))/nb_s

        elif dim==2:
            if data_file_name=='':           
                for n_q in range(nb_q) :               
                    moments[n_max,n_q]=np.mean(np.power(data,q_values[n_q]))
                for n_l in range(sp.int0(n_max-1),-1,-1):
                    nb_el=data.shape[0]
                    if nb_s==1:
                        data=(data[0:nb_el:2,0:nb_el:2]+data[0:nb_el:2,1:nb_el+1:2]+data[1:nb_el+1:2,0:nb_el:2]+data[1:nb_el+1:2,1:nb_el+1:2])/4 #Upscaling of the field
                    else:
                        data=(data[0:nb_el:2,0:nb_el:2,:]+data[0:nb_el:2,1:nb_el+1:2,:]+data[1:nb_el+1:2,0:nb_el:2,:]+data[1:nb_el+1:2,1:nb_el+1:2,:])/4 #Upscaling of the field

                    for n_q in range(nb_q) :               
                        moments[n_l,n_q]=np.mean(np.power(data,q_values[n_q])) 
            else: 
                for s in range(0,nb_s):
                    data_s=np.loadtxt(data_file_name[s],delimiter=';')
                    for n_q in range(nb_q) :               
                        moments[n_max,n_q]=moments[n_max,n_q]+np.mean(np.power(data_s,q_values[n_q]))/nb_s 
                    for n_l in range(sp.int0(n_max-1),-1,-1):
                        nb_el=data_s.shape[0]                    
                        data_s=(data_s[0:nb_el:2,0:nb_el:2]+data_s[0:nb_el:2,1:nb_el+1:2]+data_s[1:nb_el+1:2,0:nb_el:2]+data_s[1:nb_el+1:2,1:nb_el+1:2])/4 #Upscaling of the field        
                        for n_q in range(nb_q) :               
                            moments[n_l,n_q]=moments[n_l,n_q]+np.mean(np.power(data_s,q_values[n_q]))/nb_s

        np.save(file_name,moments)
   
    elif file_index==1:    # nb_ones is simply loaded
        
        moments=np.load(file_name)
        
    else: 
        print('Error in TM, file_index not equal to 0 or 1')

    # Step 2 : Evaluate K(q) for the various scaling regime (defined in l_range)
    
    x=sp.log(l_list)

    if plot_index>0:
        plt.figure(plot_index)
        # if the default value of q_values was used, the evaluation of K(q) is shown only for pre-defined values
        # to obtain a "good looking graph. Otherwise curves are plotted for all the values in q_values (in that 
        # case, less information is displayed in the legend, any all the r2 are outputted of the function)
        if q_values_bis==-1:
            p1, = plt.plot(x,sp.log(moments[:,1]),ls='None',marker='+',ms=5,mew=2,mfc='k',mec='k')
            p2, = plt.plot(x,sp.log(moments[:,9]),ls='None',marker='x',ms=5,mew=2,mfc='k',mec='k')
            p3, = plt.plot(x,sp.log(moments[:,15]),ls='None',marker='o',ms=5,mew=2,mfc='k',mec='k')
            p4, = plt.plot(x,sp.log(moments[:,19]),ls='None',marker='d',ms=5,mew=2,mfc='k',mec='k')
            p5, = plt.plot(x,sp.log(moments[:,25]),ls='None',marker='s',ms=5,mew=2,mfc='k',mec='k')
            p6, = plt.plot(x,sp.log(moments[:,30]),ls='None',marker='^',ms=5,mew=2,mfc='k',mec='k')        
            p7, = plt.plot(x,sp.log(moments[:,35]),ls='None',marker='v',ms=5,mew=2,mfc='k',mec='k')
        label_all=list()



    nb_scale_reg = len(l_range) #Evaluation of the number of scaling regime    
    
    if nb_scale_reg == 0:
        # A single scaling regime, all the resolutions are considered
        for n_q in range(nb_q) :
            y = sp.log(moments[:,n_q])
            a=sp.polyfit(x,y,1)   
            reg_lin=sp.poly1d(a)    
            Kq_1[n_q]=a[0]
            r2_1[n_q]=sp.corrcoef(x,y)[0,1]**2
            if plot_index>0:
                if q_values_bis==-1:
                    if n_q==1 or n_q==9 or n_q==15 or n_q==19 or n_q==25 or n_q==30 or n_q==35:
                        label_all.append(r'$\mathit{q}\ =\ $'+str(sp.floor(q_values[n_q]*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1[n_q]*100)/100))
                        plt.plot([x[0],x[-1]],[reg_lin(x[0]),reg_lin(x[-1])],lw=2,color='k')
                    plt.legend([p1,p2,p3,p4,p5,p6,p7],label_all,loc=2,fontsize=12,frameon=False)
                elif q_values_bis==1:
                    plt.plot(x,y,ls='None',marker='+',ms=5,mew=2,mfc='k',mec='k')                    
                    plt.plot([x[0],x[-1]],[reg_lin(x[0]),reg_lin(x[-1])],lw=2,color='k')
            Kq_2[n_q]=np.nan
            r2_2[n_q]=np.nan            
            Kq_3[n_q]=np.nan        
            r2_3[n_q]=np.nan  
    
    elif nb_scale_reg == 1:
        # A single scaling regime
        i_l_min=np.where(l_list==l_range[0][0])[0][0]
        i_l_max=np.where(l_list==l_range[0][1])[0][0]
        for n_q in range(nb_q) :
            y = sp.log(moments[:,n_q])
            a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
            reg_lin=sp.poly1d(a)        
            Kq_1[n_q]=a[0]
            r2_1[n_q]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2 
            if plot_index>0:
                if q_values_bis==-1:
                    if n_q==1 or n_q==9 or n_q==15 or n_q==19 or n_q==25 or n_q==30 or n_q==35:
                        label_all.append(r'$\mathit{q}\ =\ $'+str(sp.floor(q_values[n_q]*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1[n_q]*100)/100))
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='k')                        
                    plt.legend([p1,p2,p3,p4,p5,p6,p7],label_all,loc=2,fontsize=12,frameon=False)
                elif q_values_bis==1:
                    plt.plot(x,y,ls='None',marker='+',ms=5,mew=2,mfc='k',mec='k')                    
                    plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='k')

            # For the 2nd scaling regime
            Kq_2[n_q]=np.nan
            r2_2[n_q]=np.nan            
            # For the 3rd scaling regime
            Kq_3[n_q]=np.nan        
            r2_3[n_q]=np.nan  

    elif nb_scale_reg == 2:
        for n_q in range(nb_q) :
            y = sp.log(moments[:,n_q])
            # For the 1st scaling regime 
            i_l_min=np.where(l_list==l_range[0][0])[0][0]
            i_l_max=np.where(l_list==l_range[0][1])[0][0]
            a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
            reg_lin=sp.poly1d(a)
            Kq_1[n_q]=a[0]
            r2_1[n_q]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            if plot_index>0:
                if q_values_bis==-1:
                    if n_q==1 or n_q==9 or n_q==15 or n_q==19 or n_q==25 or n_q==30 or n_q==35:
                        label_all.append(r'$\mathit{q}\ =\ $'+str(sp.floor(q_values[n_q]*100)/100))
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='b')                        
                    plt.legend([p1,p2,p3,p4,p5,p6,p7],label_all,loc=2,fontsize=12,frameon=False)
                elif q_values_bis==1:
                    plt.plot(x,y,ls='None',marker='+',ms=5,mew=2,mfc='k',mec='k') 
                    plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='b')

            # For the 2nd scaling regime 
            i_l_min=np.where(l_list==l_range[1][0])[0][0]
            i_l_max=np.where(l_list==l_range[1][1])[0][0]
            a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
            reg_lin=sp.poly1d(a)
            Kq_2[n_q]=a[0]
            r2_2[n_q]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            if plot_index>0:
                if q_values_bis==-1:
                    if n_q==1 or n_q==9 or n_q==15 or n_q==19 or n_q==25 or n_q==30 or n_q==35:
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='r')                        
                elif q_values_bis==1:
                    plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='r')

            # For the 3rd scaling regime
            Kq_3[n_q]=np.nan        
            r2_3[n_q]=np.nan  

    elif nb_scale_reg == 3:
        for n_q in range(nb_q) :
            y = sp.log(moments[:,n_q])
            # For the 1st scaling regime 
            i_l_min=np.where(l_list==l_range[0][0])[0][0]
            i_l_max=np.where(l_list==l_range[0][1])[0][0]
            a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
            reg_lin=sp.poly1d(a)
            Kq_1[n_q]=a[0]
            r2_1[n_q]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            if plot_index>0:
                if q_values_bis==-1:
                    if n_q==1 or n_q==9 or n_q==15 or n_q==19 or n_q==25 or n_q==30 or n_q==35:
                        label_all.append(r'$\mathit{q}\ =\ $'+str(sp.floor(q_values[n_q]*100)/100))
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='b')                        
                    plt.legend([p1,p2,p3,p4,p5,p6,p7],label_all,loc=2,fontsize=12,frameon=False)
                elif q_values_bis==1:
                    plt.plot(x,y,ls='None',marker='+',ms=5,mew=2,mfc='k',mec='k') 
                    plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='b')

            # For the 2nd scaling regime 
            i_l_min=np.where(l_list==l_range[1][0])[0][0]
            i_l_max=np.where(l_list==l_range[1][1])[0][0]
            a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
            reg_lin=sp.poly1d(a)
            Kq_2[n_q]=a[0]
            r2_2[n_q]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            if plot_index>0:
                if q_values_bis==-1:
                    if n_q==1 or n_q==9 or n_q==15 or n_q==19 or n_q==25 or n_q==30 or n_q==35:
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='r') 
                elif q_values_bis==1:
                    plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='r')

            # For the 3rd scaling regime 
            i_l_min=np.where(l_list==l_range[2][0])[0][0]
            i_l_max=np.where(l_list==l_range[2][1])[0][0]
            a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)  
            reg_lin=sp.poly1d(a)
            Kq_3[n_q]=a[0]
            r2_3[n_q]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
            if plot_index>0:
                if q_values_bis==-1:
                    if n_q==1 or n_q==9 or n_q==15 or n_q==19 or n_q==25 or n_q==30 or n_q==35:
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='g') 
                elif q_values_bis==1:
                    plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='g')


    else :
        print('Error in TM, l_range has wrong size')
        
    if plot_index>0:
        plt.xlabel(r'$\log(\lambda)$',fontsize=20,color='k')
        plt.ylabel(r'$\log(TM_\lambda)$',fontsize=20,color='k')
        plt.title('TM Analysis',fontsize=20,color='k')  
        ax=plt.gca()
        for xtick in ax.get_xticklabels():
            plt.setp(xtick,fontsize=14)
        for ytick in ax.get_yticklabels():
            plt.setp(ytick,fontsize=14)    
        plt.savefig('evaluation_K_q_in_TM.png')
    
    
    if plot_index>0:
        plt.figure(plot_index+1)
        if nb_scale_reg == 0:
            p1,=plt.plot(q_values,Kq_1,lw=2,color='k')
        elif nb_scale_reg == 1:
            p1,=plt.plot(q_values,Kq_1,lw=2,color='k')
        elif nb_scale_reg == 2:
            p1,=plt.plot(q_values,Kq_1,lw=2,color='b')
            p2,=plt.plot(q_values,Kq_2,lw=2,color='r')  
            plt.legend([p1,p2],['['+str(l_range[0][0])+','+str(l_range[0][1])+']','['+str(l_range[1][0])+','+str(l_range[1][1])+']'],loc=2,frameon=False)
        elif nb_scale_reg == 3:
            p1,=plt.plot(q_values,Kq_1,lw=2,color='b')
            p2,=plt.plot(q_values,Kq_2,lw=2,color='r')
            p3,=plt.plot(q_values,Kq_3,lw=2,color='g')
            plt.legend([p1,p2,p3],['['+str(l_range[0][0])+','+str(l_range[0][1])+']','['+str(l_range[1][0])+','+str(l_range[1][1])+']','['+str(l_range[2][0])+','+str(l_range[2][1])+']'],loc=2,frameon=False)
        else :  
            print('Error in TM, l_range has wrong size')

        plt.xlabel(r'$\mathit{q}$',fontsize=20,color='k')
        plt.ylabel(r'$\mathit{K(q)}$',fontsize=20,color='k')
        #plt.title('TM Analysis',fontsize=20,color='k')  
        ax=plt.gca()
        ax.set_xlim([q_values[0], q_values[-1]])        
        for xtick in ax.get_xticklabels():
            plt.setp(xtick,fontsize=14)
        for ytick in ax.get_yticklabels():
            plt.setp(ytick,fontsize=14)        
    
        plt.savefig('K_q_in_TM.png')
    
    return Kq_1,Kq_2,Kq_3,r2_1,r2_2,r2_3    
    

############################################################################################
############################################################################################

# This script allows to perform a DTM analysis. 
# The following options are available : 
#      - 1D or 2D analysis
#      - Sample analysis or average analysis on sets of samples
#      - Possibility to take into account up to three scaling regime
#      - Possibility to display or not the graphics

def DTM(data,q_values,data_file_name,dim,l_range,DTM_index,file_index,file_name,plot_index):

    # Inputs :
    # - data : data is a numpy matrix that contain the data to be analysed. 
    #          Its meaning depends on dim
    #      - dim=1 : 1D analysis are performed
    #                data is a 2D matrix, each column is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                if one column, then a sample analysis is performed)
    #      - dim=2 : 2D analysis are performed
    #                data is a 3D matrix, each layer (i.e. data[:,:,s]) is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                (if one layer, then a sample analysis is performed)
    # - q_values : a numpy vector containing the different q for which K(q,eta) is evaluated
    #                  usually q=np.array([1.5])
    #                  if q contains more than one term, then UM parameters are estimated for each q, and the 
    #                  mean and standard deviation are provided
    # - data_file_name : - if data_file_name =='' then "data" is used for the analysis
    #                    - otherwise : data_file_name is list of .csv file name
    #                                  the 1D or 2D field contained in each file is considered as a independant sample
    #                                  (used for large files)                
    # - l_range is a list of list. Each element contains two elements (reso_min and reso_max)
    #                and a fractal dimension is evaluated between reso_min and reso_max
    #                if l_range=[] then a single scaling regime is considered and the all the available resolution are used
    #                the max length of l_range is 3; ie no more than three scaling regime can be studied
    # - DTM_index : 1 --> standard DTM analysis (Lavallee, 1993)
    #               2 --> modified DTM analysis (Veneziano, 1999)
    # - file_index : the computation of the moments might be quite long, and therefore can be recorded in a file 
    #                that can be used later
    #                file_index=0 --> "moments" is recorded
    #                file_index=1 --> "moments" is retrieved from an existing file
    # - file_name : the name of the file recorded or retrieved
    #               WARNING the file should have a .npy extension !!
    # - plot_index : the number of the first figure opened for graph display 
    #
    # Outputs : 
    # - UM_par1 : a vector containing UM parameter estimates (and if more than one q estimates of the uncertainty on the various q)
    #             for the 1st scaling regime defined in l_range[0]
    #                UM_par1 = [alpha,C1,sdt_alpha,sdt_C1]
    # - UM_par2 : same as UM_par_1 but for the 2nd scaling regime (defined in l_range[1])
    #                             UM_par2 --> large scale
    # - UM_par2 : same as UM_par_1 but for the 2nd scaling regime (defined in l_range[2])
    #                             UM_par2 --> large scale
    #
    # Note 1 : the output of "unused" scaling regime(s) are returned as "nan"  
    # Note 2 : a ration of 2 is used in the upscaling proces
    
    
    # Evaluating l_max (maximum resolution) and nb_s (number of samples)
    if dim ==1:
        if data_file_name=='':
            l_max=data.shape[0]
            if len(data.shape)==1:
                nb_s=1
            else :    
                nb_s=data.shape[1]
        else:
            nb_s=len(data_file_name)
            l_max=np.loadtxt(data_file_name[0]).shape[0]
    elif dim==2:
        if data_file_name=='':
            l_max=data.shape[0]
            if len(data.shape)==2:
                nb_s=1
            else:
                nb_s=data.shape[2]
        else:
            nb_s=len(data_file_name)
            l_max=np.loadtxt(data_file_name[0],delimiter=';').shape[0]
    else:
        print('Error in TM, wrong dim')
  
    # TO BE DONE : write an error message of the size is not a power of 2
    n_max=sp.log(l_max)/sp.log(2)
    l_list=np.zeros((n_max+1))    
    for n in range(sp.int0(n_max)+1):
        l_list[n]=sp.power(2,n)

    # q_values
    nb_q=q_values.shape[0]
     
    # eta_values
    logeta=np.linspace(-2,1,num=34)
    eta_values=10.**logeta
    nb_eta=eta_values.shape[0]
    
    moments=np.zeros((n_max+1,nb_q,nb_eta))
    K_qeta_1=np.zeros((nb_q,nb_eta))
    K_qeta_2=np.zeros((nb_q,nb_eta))
    K_qeta_3=np.zeros((nb_q,nb_eta))
    r2_1=np.zeros((nb_q,nb_eta))
    r2_2=np.zeros((nb_q,nb_eta))
    r2_3=np.zeros((nb_q,nb_eta))       
    UM_par_1=list()
    UM_par_2=list()
    UM_par_3=list()

    #########
    ## Step 1 : Evaluate the DTM moments
    ##          Results are stored in a numpy array moments 
    ##                               (1st index --> resolution,)
    ##                               (2nd index --> value of q, same order as in q_values)
    ##                               (3nd index --> value of eta, same order as in eta_values)
    #########                               
    
    if file_index == 0:  # moments is computed
        if dim == 1 :
            if data_file_name == '' :           
                if DTM_index == 1 :
                    for n_eta in range(nb_eta) :
                        data_eta=np.power(data,eta_values[n_eta])
                        for n_q in range(nb_q) :               
                            moments[n_max,n_q,n_eta]=np.mean(np.power(data_eta,q_values[n_q]))
                        for n_l in range(sp.int0(n_max-1),-1,-1):
                            nb_el=data_eta.shape[0]
                            data_eta=(data_eta[0:nb_el:2,:]+data_eta[1:nb_el+1:2,:])/2 # Upscaling of the field
                            for n_q in range(nb_q) :               
                                moments[n_l,n_q,n_eta]=np.mean(np.power(data_eta,q_values[n_q]))
                elif DTM_index == 2 : 
                    for n_eta in range(nb_eta) : 
                        for n_q in range(nb_q) :  
                            moments[n_max,n_q,n_eta]=np.mean(np.power(data,q_values[n_q]*eta_values[n_eta]))/np.power(np.mean(np.power(data,eta_values[n_eta])),q_values[n_q])
                    for n_l in range(sp.int0(n_max-1),-1,-1):
                        nb_el=data.shape[0]
                        data=(data[0:nb_el:2,:]+data[1:nb_el+1:2,:])/2 # Upscaling of the field
                        for n_eta in range(nb_eta) :                        
                            for n_q in range(nb_q) :               
                                moments[n_l,n_q,n_eta]=np.mean(np.power(data,q_values[n_q]*eta_values[n_eta]))/np.power(np.mean(np.power(data,eta_values[n_eta])),q_values[n_q])
            else: 
                if DTM_index == 1 :                
                    for s in range(0,nb_s):
                        data_s=np.loadtxt(data_file_name[s],delimiter=';')                    
                        for n_eta in range(nb_eta) :
                            data_eta=np.power(data_s,eta_values[n_eta])
                            for n_q in range(nb_q) :               
                                moments[n_max,n_q,n_eta]=moments[n_max,n_q,n_eta]+np.mean(np.power(data_eta,q_values[n_q]))/nb_s
                            for n_l in range(sp.int0(n_max-1),-1,-1):
                                nb_el=data_eta.shape[0]
                                data_eta=(data_eta[0:nb_el:2]+data_eta[1:nb_el+1:2])/2 # Upscaling of the field
                                for n_q in range(nb_q) :               
                                    moments[n_l,n_q,n_eta]=moments[n_l,n_q,n_eta]+np.mean(np.power(data_eta,q_values[n_q]))/nb_s
                elif DTM_index == 2 : 
                    moments2=np.zeros((n_max+1,nb_q,nb_eta))
                    for s in range(0,nb_s):
                        data_s=np.loadtxt(data_file_name[s],delimiter=';') 
                        for n_eta in range(nb_eta) : 
                            for n_q in range(nb_q) :  
                                moments[n_max,n_q,n_eta]=moments[n_max,n_q,n_eta]+np.mean(np.power(data_s,q_values[n_q]*eta_values[n_eta]))/nb_s
                                moments2[n_max,n_q,n_eta]=moments2[n_max,n_q,n_eta]+np.mean(np.power(data_s,eta_values[n_eta]))/nb_s
                        for n_l in range(sp.int0(n_max-1),-1,-1):
                            nb_el=data_s.shape[0]
                            data_s=(data_s[0:nb_el:2]+data_s[1:nb_el+1:2])/2 # Upscaling of the field
                            for n_eta in range(nb_eta) :                        
                                for n_q in range(nb_q) :               
                                    moments[n_l,n_q,n_eta]=moments[n_l,n_q,n_eta]+np.mean(np.power(data_s,q_values[n_q]*eta_values[n_eta]))/nb_s
                                    moments2[n_l,n_q,n_eta]=moments2[n_l,n_q,n_eta]+np.mean(np.power(data_s,eta_values[n_eta]))/nb_s
                    for n_q in range(nb_q) : 
                        moments[:,n_q,:]=moments[:,n_q,:]/np.power(moments2[:,n_q,:],q_values[n_q])
        elif dim==2 :
            if data_file_name == '' :           
                if DTM_index == 1 :
                    for n_eta in range(nb_eta) :
                        data_eta=np.power(data,eta_values[n_eta])
                        for n_q in range(nb_q) :               
                            moments[n_max,n_q,n_eta]=np.mean(np.power(data_eta,q_values[n_q]))
                        for n_l in range(sp.int0(n_max-1),-1,-1):
                            nb_el=data_eta.shape[0]
                            if nb_s==1:
                                data_eta=(data_eta[0:nb_el:2,0:nb_el:2]+data_eta[0:nb_el:2,1:nb_el+1:2]+data_eta[1:nb_el+1:2,0:nb_el:2]+data_eta[1:nb_el+1:2,1:nb_el+1:2])/4 #Upscaling of the field
                            else:
                                data_eta=(data_eta[0:nb_el:2,0:nb_el:2,:]+data_eta[0:nb_el:2,1:nb_el+1:2,:]+data_eta[1:nb_el+1:2,0:nb_el:2,:]+data_eta[1:nb_el+1:2,1:nb_el+1:2,:])/4 #Upscaling of the field 
                            for n_q in range(nb_q) :               
                                moments[n_l,n_q,n_eta]=np.mean(np.power(data_eta,q_values[n_q]))
                elif DTM_index == 2 : 
                    for n_eta in range(nb_eta) : 
                        for n_q in range(nb_q) :  
                            moments[n_max,n_q,n_eta]=np.mean(np.power(data,q_values[n_q]*eta_values[n_eta]))/np.power(np.mean(np.power(data,eta_values[n_eta])),q_values[n_q])
                    for n_l in range(sp.int0(n_max-1),-1,-1):
                        nb_el=data.shape[0]
                        if nb_s==1:
                            data=(data[0:nb_el:2,0:nb_el:2]+data[0:nb_el:2,1:nb_el+1:2]+data[1:nb_el+1:2,0:nb_el:2]+data[1:nb_el+1:2,1:nb_el+1:2])/4 #Upscaling of the field
                        else:
                            data=(data[0:nb_el:2,0:nb_el:2,:]+data[0:nb_el:2,1:nb_el+1:2,:]+data[1:nb_el+1:2,0:nb_el:2,:]+data[1:nb_el+1:2,1:nb_el+1:2,:])/4 #Upscaling of the field 
                        for n_eta in range(nb_eta) :                        
                            for n_q in range(nb_q) :               
                                moments[n_l,n_q,n_eta]=np.mean(np.power(data,q_values[n_q]*eta_values[n_eta]))/np.power(np.mean(np.power(data,eta_values[n_eta])),q_values[n_q])
            else: 
                if DTM_index == 1 :                
                    for s in range(0,nb_s):
                        data_s=np.loadtxt(data_file_name[s],delimiter=';')                    
                        for n_eta in range(nb_eta) :
                            data_eta=np.power(data_s,eta_values[n_eta])
                            for n_q in range(nb_q) :               
                                moments[n_max,n_q,n_eta]=moments[n_max,n_q,n_eta]+np.mean(np.power(data_eta,q_values[n_q]))/nb_s
                            for n_l in range(sp.int0(n_max-1),-1,-1):
                                nb_el=data_eta.shape[0]
                                data_eta=(data_eta[0:nb_el:2,0:nb_el:2]+data_eta[0:nb_el:2,1:nb_el+1:2]+data_eta[1:nb_el+1:2,0:nb_el:2]+data_eta[1:nb_el+1:2,1:nb_el+1:2])/4 #Upscaling of the field
                                for n_q in range(nb_q) :               
                                    moments[n_l,n_q,n_eta]=moments[n_l,n_q,n_eta]+np.mean(np.power(data_eta,q_values[n_q]))/nb_s
                elif DTM_index == 2 : 
                    moments2=np.zeros((n_max+1,nb_q,nb_eta))
                    for s in range(0,nb_s):
                        data_s=np.loadtxt(data_file_name[s],delimiter=';') 
                        for n_eta in range(nb_eta) : 
                            for n_q in range(nb_q) :  
                                moments[n_max,n_q,n_eta]=moments[n_max,n_q,n_eta]+np.mean(np.power(data_s,q_values[n_q]*eta_values[n_eta]))/nb_s
                                moments2[n_max,n_q,n_eta]=moments2[n_max,n_q,n_eta]+np.mean(np.power(data_s,eta_values[n_eta]))/nb_s
                        for n_l in range(sp.int0(n_max-1),-1,-1):
                            nb_el=data_s.shape[0]
                            data_s=(data_s[0:nb_el:2,0:nb_el:2]+data_s[0:nb_el:2,1:nb_el+1:2]+data_s[1:nb_el+1:2,0:nb_el:2]+data_s[1:nb_el+1:2,1:nb_el+1:2])/4 #Upscaling of the field                                    
                            for n_eta in range(nb_eta) :                        
                                for n_q in range(nb_q) :               
                                    moments[n_l,n_q,n_eta]=moments[n_l,n_q,n_eta]+np.mean(np.power(data_s,q_values[n_q]*eta_values[n_eta]))/nb_s
                                    moments2[n_l,n_q,n_eta]=moments2[n_l,n_q,n_eta]+np.mean(np.power(data_s,eta_values[n_eta]))/nb_s
                    for n_q in range(nb_q) : 
                        moments[:,n_q,:]=moments[:,n_q,:]/np.power(moments2[:,n_q,:],q_values[n_q])

        np.save(file_name,moments)
   
    elif file_index==1:    # nb_ones is simply loaded
        moments=np.load(file_name)
        
    else: 
        print('Error in fTM, file_index not equal to 0 or 1')

    
    #########
    ## Step 2 : Evaluate K(q) for the various scaling regime (defined in l_range)
    #########
    
    nb_scale_reg = len(l_range) #Evaluation of the number of scaling regime     
    
    # x is the vector of the resolutions 
    x=sp.log(l_list)    
    
    # Defintion of the list of markers
    list_markers=['+','x','o','d']    


    #
    # Evaulation of K_qeta for the various scaling regimes
    #
    
    if nb_scale_reg == 0:
        # A single scaling regime, all the resolutions are considered
        for n_q in range(nb_q) :
  
            if plot_index>0:
                plt.figure(plot_index+n_q)
                ind_curve=0
            
            for n_eta in range(nb_eta) :
                y = sp.log(moments[:,n_q,n_eta])
                a=sp.polyfit(x,y,1)   
                reg_lin=sp.poly1d(a)    
                K_qeta_1[n_q,n_eta]=a[0]
                r2_1[n_q,n_eta]=sp.corrcoef(x,y)[0,1]**2
                if plot_index>0:
                    if n_eta==21 or n_eta==23 or n_eta==25 or n_eta==27 : 
                        plt.plot(x,y,ls='None',marker=list_markers[ind_curve],ms=5,mew=2,mfc='k',mec='k',label=r'$\eta\ =\ $'+str(sp.floor(eta_values[n_eta]*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1[n_q,n_eta]*100)/100))                        
                        plt.plot([x[0],x[-1]],[reg_lin(x[0]),reg_lin(x[-1])],lw=2,color='k')
                        ind_curve=ind_curve+1
                K_qeta_2[n_q,n_eta]=np.nan
                K_qeta_3[n_q,n_eta]=np.nan        
            
            if plot_index>0:
                plt.xlabel(r'$\log(\lambda)$',fontsize=20,color='k')
                plt.ylabel(r'$\log(DTM_\lambda)$',fontsize=20,color='k')
                plt.title('DTM Analysis, q = '+str(q_values[n_q]),fontsize=20,color='k')  
                plt.legend(loc=2,fontsize=12,frameon=False)                
                ax=plt.gca()
                for xtick in ax.get_xticklabels():
                    plt.setp(xtick,fontsize=14)
                    for ytick in ax.get_yticklabels():
                        plt.setp(ytick,fontsize=14)    
                plt.savefig('DTM_test_q_'+str(q_values[n_q])+'.png')
        
    elif nb_scale_reg == 1 :
        
        # For the 1st scaling regime
        # Retrieving the indexes of the corresponding resolutions 
        i_l_min=np.where(l_list==l_range[0][0])[0][0]
        i_l_max=np.where(l_list==l_range[0][1])[0][0]
        # Computing K_qeta        
        for n_q in range(nb_q) :
            if plot_index>0:
                plt.figure(plot_index+n_q)
                ind_curve=0    
            for n_eta in range(nb_eta) :
                y = sp.log(moments[:,n_q,n_eta])
                a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)   
                reg_lin=sp.poly1d(a)    
                K_qeta_1[n_q,n_eta]=a[0]
                r2_1[n_q,n_eta]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
                if plot_index>0:
                    if n_eta==21 or n_eta==23 or n_eta==25 or n_eta==27 : 
                        plt.plot(x,y,ls='None',marker=list_markers[ind_curve],ms=5,mew=2,mfc='k',mec='k',label=r'$\eta\ =\ $'+str(sp.floor(eta_values[n_eta]*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1[n_q,n_eta]*100)/100))                        
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='k')
                        ind_curve=ind_curve+1
                K_qeta_2[n_q,n_eta]=np.nan
                K_qeta_3[n_q,n_eta]=np.nan            
            if plot_index>0:
                plt.xlabel(r'$\log(\lambda)$',fontsize=20,color='k')
                plt.ylabel(r'$\log(DTM_\lambda)$',fontsize=20,color='k')
                plt.title('DTM Analysis, q = '+str(q_values[n_q]),fontsize=20,color='k')  
                plt.legend(loc=2,fontsize=12,frameon=False)                
                ax=plt.gca()
                for xtick in ax.get_xticklabels():
                    plt.setp(xtick,fontsize=14)
                    for ytick in ax.get_yticklabels():
                        plt.setp(ytick,fontsize=14)    
                plt.savefig('DTM_test_q_'+str(q_values[n_q])+'.png')
        
    elif nb_scale_reg == 2 :
        # Computing K_qeta        
        for n_q in range(nb_q) :  
            if plot_index>0:
                plt.figure(plot_index+n_q)
                ind_curve=0            
            for n_eta in range(nb_eta) :
                y = sp.log(moments[:,n_q,n_eta])
                # For the 1st scaling regime
                i_l_min=np.where(l_list==l_range[0][0])[0][0]
                i_l_max=np.where(l_list==l_range[0][1])[0][0]
                a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)   
                reg_lin=sp.poly1d(a)    
                K_qeta_1[n_q,n_eta]=a[0]
                r2_1[n_q,n_eta]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
                if plot_index>0:
                    if n_eta==21 or n_eta==23 or n_eta==25 or n_eta==27 :                       
                        plt.plot(x,y,ls='None',marker=list_markers[ind_curve],ms=5,mew=2,mfc='k',mec='k',label=r'$\eta\ =\ $'+str(sp.floor(eta_values[n_eta]*100)/100))                        
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='b')
                        ind_curve=ind_curve+1
                
                # For the 2nd scaling regime
                i_l_min=np.where(l_list==l_range[1][0])[0][0]
                i_l_max=np.where(l_list==l_range[1][1])[0][0]
                a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)   
                reg_lin=sp.poly1d(a)    
                K_qeta_2[n_q,n_eta]=a[0]
                r2_2[n_q,n_eta]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
                if plot_index>0:
                    if n_eta==21 or n_eta==23 or n_eta==25 or n_eta==27 : 
                        #plt.plot(x,y,ls='None',marker=list_markers[ind_curve],ms=5,mew=2,mfc='k',mec='k',label=r'$\eta\ =\ $'+str(sp.floor(eta_values[n_eta]*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1[n_q,n_eta]*100)/100))                        
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='r')

                # For the 3rd scaling regime
                K_qeta_3[n_q,n_eta]=np.nan
                     
            if plot_index>0:
                plt.xlabel(r'$\log(\lambda)$',fontsize=20,color='k')
                plt.ylabel(r'$\log(DTM_\lambda)$',fontsize=20,color='k')
                plt.title('DTM Analysis, q = '+str(q_values[n_q]),fontsize=20,color='k')  
                plt.legend(loc=2,fontsize=12,frameon=False)                
                ax=plt.gca()
                for xtick in ax.get_xticklabels():
                    plt.setp(xtick,fontsize=14)
                    for ytick in ax.get_yticklabels():
                        plt.setp(ytick,fontsize=14)    
                plt.savefig('DTM_test_q_'+str(q_values[n_q])+'.png')
        

    elif nb_scale_reg == 3 :
       
        for n_q in range(nb_q) :
            if plot_index>0:
                plt.figure(plot_index+n_q)
                ind_curve=0    
            for n_eta in range(nb_eta) :
                y = sp.log(moments[:,n_q,n_eta])
                # For the 1st scaling regime
                i_l_min=np.where(l_list==l_range[0][0])[0][0]
                i_l_max=np.where(l_list==l_range[0][1])[0][0]
                a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)   
                reg_lin=sp.poly1d(a)    
                K_qeta_1[n_q,n_eta]=a[0]
                r2_1[n_q,n_eta]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
                if plot_index>0:
                    if n_eta==21 or n_eta==23 or n_eta==25 or n_eta==27 : 
                        plt.plot(x,y,ls='None',marker=list_markers[ind_curve],ms=5,mew=2,mfc='k',mec='k',label=r'$\eta\ =\ $'+str(sp.floor(eta_values[n_eta]*100)/100))                        
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='b')
                        ind_curve=ind_curve+1        
                # For the 2nd scaling regime
                i_l_min=np.where(l_list==l_range[1][0])[0][0]
                i_l_max=np.where(l_list==l_range[1][1])[0][0]
                a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)   
                reg_lin=sp.poly1d(a)    
                K_qeta_2[n_q,n_eta]=a[0]
                r2_2[n_q,n_eta]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
                if plot_index>0:
                    if n_eta==21 or n_eta==23 or n_eta==25 or n_eta==27 : 
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='r')
                # For the 3rd scaling regime
                i_l_min=np.where(l_list==l_range[2][0])[0][0]
                i_l_max=np.where(l_list==l_range[2][1])[0][0]
                a=sp.polyfit(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1],1)   
                reg_lin=sp.poly1d(a)    
                K_qeta_3[n_q,n_eta]=a[0]
                r2_3[n_q,n_eta]=sp.corrcoef(x[i_l_min:i_l_max+1],y[i_l_min:i_l_max+1])[0,1]**2
                if plot_index>0:
                    if n_eta==21 or n_eta==23 or n_eta==25 or n_eta==27 : 
                        plt.plot([x[i_l_min],x[i_l_max]],[reg_lin(x[i_l_min]),reg_lin(x[i_l_max])],lw=2,color='g')
            
            if plot_index>0:
                plt.xlabel(r'$\log(\lambda)$',fontsize=20,color='k')
                plt.ylabel(r'$\log(DTM_\lambda)$',fontsize=20,color='k')
                plt.title('DTM Analysis, q = '+str(q_values[n_q]),fontsize=20,color='k')  
                plt.legend(loc=2,fontsize=12,frameon=False)                
                ax=plt.gca()
                for xtick in ax.get_xticklabels():
                    plt.setp(xtick,fontsize=14)
                    for ytick in ax.get_yticklabels():
                        plt.setp(ytick,fontsize=14)    
                plt.savefig('DTM_test_q_'+str(q_values[n_q])+'.png')


    else :      
        print('Error in DTM, K_qeta, l_range has wrong size')

    
    ###########
    ## Step 3 : Evaluation of UM parameters
    ###########

    if nb_scale_reg == 0 or nb_scale_reg == 1 or nb_scale_reg ==2 or nb_scale_reg == 3 : 
        # Evaluation of UM_par_1 
        alpha_q=np.zeros((nb_q,))        
        C1_q=np.zeros((nb_q,))        
        if plot_index>0: 
            plt.figure(plot_index+nb_q+1)
        for n_q in range(nb_q) : 
            alpha_buf=np.zeros((nb_eta,))
            x_buf = np.log10(eta_values)
            y_buf = np.log10(K_qeta_1[n_q,:])
            for n_eta in range(15,nb_eta-4):
                a=sp.polyfit(x_buf[n_eta-2:n_eta+3],y_buf[n_eta-2:n_eta+3],1)
                alpha_buf[n_eta]=a[0]
            ind=np.argmax(alpha_buf)
            a=sp.polyfit(x_buf[ind-2:ind+3],y_buf[ind-2:ind+3],1)
            alpha_q[n_q]=a[0]
            C1_q[n_q]=np.power(10,a[1])*(a[0]-1)/(q_values[n_q]**a[0]-q_values[n_q])
            if plot_index>0:
                plt.plot(x_buf,y_buf,ls='None',marker=list_markers[n_q],ms=5,mew=2,mfc='k',mec='k',label='q = '+str(q_values[n_q])+'; '+r'$\alpha\ =\ $'+str(sp.floor(alpha_q[n_q]*100)/100)+'; '+r'$\mathit{C}_1\ =\ $'+str(sp.floor(C1_q[n_q]*100)/100))
                reg_lin=sp.poly1d(a)
                plt.plot([x_buf[ind-2],x_buf[ind+2]],[reg_lin(x_buf[ind-2]),reg_lin(x_buf[ind+2])],lw=2,color='k')                        
        if plot_index>0:
            plt.xlabel(r'$\log_{10}(\eta)$',fontsize=20,color='k')
            plt.ylabel(r'$\log_{10}\mathit{K}(\mathit{q},\eta)$',fontsize=20,color='k')
            plt.title('UM paramaters for 1st scaling regime',fontsize=20,color='k')  
            plt.legend(loc=2,fontsize=12,frameon=False)            
            ax=plt.gca()
            for xtick in ax.get_xticklabels():
                plt.setp(xtick,fontsize=14)
            for ytick in ax.get_yticklabels():
                    plt.setp(ytick,fontsize=14)    
            plt.savefig('Determination_UM_par_1.png')            
        alpha_1=np.mean(alpha_q)
        C1_1=np.mean(C1_q)
        if nb_q>1:
            std_alpha_1=np.std(alpha_q)
            std_C1_1=np.std(C1_q)
        else :
            std_alpha_1=np.nan
            std_C1_1=np.nan
        UM_par_1=[alpha_1,C1_1,std_alpha_1,std_C1_1]        
        UM_par_2=[np.nan,np.nan,np.nan,np.nan]
        UM_par_3=[np.nan,np.nan,np.nan,np.nan]

    if nb_scale_reg ==2 or nb_scale_reg == 3 :         
        # Evaluation of UM_par_2
        alpha_q=np.zeros((nb_q,))        
        C1_q=np.zeros((nb_q,))        
        if plot_index>0: 
            plt.figure(plot_index+nb_q+2)
        for n_q in range(nb_q) : 
            alpha_buf=np.zeros((nb_eta,))
            x_buf = np.log10(eta_values)
            y_buf = np.log10(K_qeta_2[n_q,:])
            for n_eta in range(15,nb_eta-4):
                a=sp.polyfit(x_buf[n_eta-2:n_eta+3],y_buf[n_eta-2:n_eta+3],1)
                alpha_buf[n_eta]=a[0]
            ind=np.argmax(alpha_buf)
            a=sp.polyfit(x_buf[ind-2:ind+3],y_buf[ind-2:ind+3],1)
            alpha_q[n_q]=a[0]
            C1_q[n_q]=10**a[1]*(a[0]-1)/(q_values[n_q]**a[0]-q_values[n_q])
            if plot_index>0:
                plt.plot(x_buf,y_buf,ls='None',marker=list_markers[n_q],ms=5,mew=2,mfc='k',mec='k',label='q = '+str(q_values[n_q])+'; '+r'$\alpha\ =\ $'+str(sp.floor(alpha_q[n_q]*100)/100)+'; '+r'$\mathit{C}_1\ =\ $'+str(sp.floor(C1_q[n_q]*100)/100))
                reg_lin=sp.poly1d(a)
                plt.plot([x_buf[ind-2],x_buf[ind+2]],[reg_lin(x_buf[ind-2]),reg_lin(x_buf[ind+2])],lw=2,color='k')                        
        if plot_index>0:
            plt.xlabel(r'$\log_{10}(\eta)$',fontsize=20,color='k')
            plt.ylabel(r'$\log_{10}\mathit{K}(\mathit{q},\eta)$',fontsize=20,color='k')
            plt.title('UM paramaters for 2nd scaling regime',fontsize=20,color='k')  
            plt.legend(loc=2,fontsize=12,frameon=False)            
            ax=plt.gca()
            for xtick in ax.get_xticklabels():
                plt.setp(xtick,fontsize=14)
            for ytick in ax.get_yticklabels():
                    plt.setp(ytick,fontsize=14)    
            plt.savefig('Determination_UM_par_2.png')            
        alpha_2=np.mean(alpha_q)
        C1_2=np.mean(C1_q)
        if nb_q>1:
            std_alpha_2=np.std(alpha_q)
            std_C1_2=np.std(C1_q)
        else :
            std_alpha_2=np.nan
            std_C1_2=np.nan        
        UM_par_2=[alpha_2,C1_2,std_alpha_2,std_C1_2]
        UM_par_3=[np.nan,np.nan,np.nan,np.nan]

    if  nb_scale_reg == 3 :         
        # Evaluation of UM_par_3
        alpha_q=np.zeros((nb_q,))        
        C1_q=np.zeros((nb_q,))        
        if plot_index>0: 
            plt.figure(plot_index+nb_q+3)
        for n_q in range(nb_q) : 
            alpha_buf=np.zeros((nb_eta,))
            x_buf = np.log10(eta_values)
            y_buf = np.log10(K_qeta_3[n_q,:])
            for n_eta in range(15,nb_eta-4):
                a=sp.polyfit(x_buf[n_eta-2:n_eta+3],y_buf[n_eta-2:n_eta+3],1)
                alpha_buf[n_eta]=a[0]
            ind=np.argmax(alpha_buf)
            a=sp.polyfit(x_buf[ind-2:ind+3],y_buf[ind-2:ind+3],1)
            alpha_q[n_q]=a[0]
            C1_q[n_q]=10**a[1]*(a[0]-1)/(q_values[n_q]**a[0]-q_values[n_q])
            if plot_index>0:
                plt.plot(x_buf,y_buf,ls='None',marker=list_markers[n_q],ms=5,mew=2,mfc='k',mec='k',label='q = '+str(q_values[n_q])+'; '+r'$\alpha\ =\ $'+str(sp.floor(alpha_q[n_q]*100)/100)+'; '+r'$\mathit{C}_1\ =\ $'+str(sp.floor(C1_q[n_q]*100)/100))
                reg_lin=sp.poly1d(a)
                plt.plot([x_buf[ind-2],x_buf[ind+2]],[reg_lin(x_buf[ind-2]),reg_lin(x_buf[ind+2])],lw=2,color='k')                        
        if plot_index>0:
            plt.xlabel(r'$\log_{10}(\eta)$',fontsize=20,color='k')
            plt.ylabel(r'$\log_{10}\mathit{K}(\mathit{q},\eta)$',fontsize=20,color='k')
            plt.title('UM paramaters for 3rd scaling regime',fontsize=20,color='k')  
            plt.legend(loc=2,fontsize=12,frameon=False)            
            ax=plt.gca()
            for xtick in ax.get_xticklabels():
                plt.setp(xtick,fontsize=14)
            for ytick in ax.get_yticklabels():
                    plt.setp(ytick,fontsize=14)    
            plt.savefig('Determination_UM_par_3.png')            
        alpha_3=np.mean(alpha_q)
        C1_3=np.mean(C1_q)
        if nb_q>1:
            std_alpha_3=np.std(alpha_q)
            std_C1_3=np.std(C1_q)
        else :
            std_alpha_3=np.nan
            std_C1_3=np.nan        
        UM_par_3=[alpha_3,C1_3,std_alpha_3,std_C1_3]
        
    return UM_par_1,UM_par_2,UM_par_3    
    


############################################################################################
############################################################################################  
###  Tools for UM
############################################################################################
############################################################################################  

##########
# Theoretical formula for K(q)
##########
def  K_q (q_values,alpha,C1,H) :
    # The output is the values of the scaling moment function
    k_q_theo=np.zeros(q_values.shape)
    if alpha==1 :
        for i in range(q_values.shape[0]) :
            k_q_theo[i]=C1*q_values[i]*np.log(q_values[i])+H*q_values[i]
    else :
        for i in range(q_values.shape[0]) :
            k_q_theo[i]=C1*(q_values[i]**alpha-q_values[i])/(alpha-1)+H*q_values[i]

    return k_q_theo



##########
# Theoretical formula for c(gamma)
##########

def c_gamma (g_values,alpha,C1,H) :
    #The output is the values of the co-dimension function

    c_g_theo=np.zeros(g_values.shape[0])
    if alpha<1 :
       alphap=alpha/(alpha-1);
       gamma0=-C1*alphap/alpha
       for i in range (g_values.shape[0]) :
           if g_values[i]<=gamma0 :
               if (g_values[i]-H)/(C1*alphap)+1/alpha==0 :
                   c_g_theo[i]=np.nan
               else :
                   c_g_theo[i]=C1*((g_values[i]-H)/(C1*alphap)+1/alpha)**alphap
           else :
               c_g_theo[i]=np.nan
     
    elif alpha==1 :
        for i in range (g_values.shape[0]) :
            c_g_theo[i]=C1*np.exp((g_values(i)-H)/C1-1)
    
    else :
        alphap=alpha/(alpha-1);
        gamma0=-C1*alphap/alpha;
        for i in range (g_values.shape[0]) :
            if g_values[i]<=gamma0 :
                c_g_theo[i]=0
            else :
                c_g_theo[i]=C1*((g_values[i]-H)/(C1*alphap)+1/alpha)**alphap;
        
    return c_g_theo


##########
# Assessement of qs, gammas 
##########


def assess_qs (alpha,C1) :
    qs = (1/C1)**(1/alpha)
    return qs

def assess_gammas (alpha,C1) : 
    gammas=C1*(alpha/(alpha-1))*((1/C1)**((alpha-1)/alpha)-1/alpha)
    return gammas


##########
# Assessement of qD
##########


def assess_qD (alpha,C1,D,plot_index) :
    from scipy.optimize import fsolve
    def f (x):
        return C1*(x**alpha-x)/((x-1)*D*(alpha-1))-1
    qD=fsolve(f,1.0001)

    if plot_index>0:
        plt.figure(plot_index)
        q_values=np.arange(1,20001,1)/100
        nb_q=q_values.shape[0]
        y1=np.zeros((nb_q,))
        y2=np.zeros((nb_q,))        
        for i in range(nb_q):
            q=q_values[i]
            y1[i]=C1*(q**alpha-q)/(alpha-1)-(q-1)*D
        plt.plot(q_values,y1,lw=2,color='k')
        plt.plot(q_values,y2,lw=2,color='k')
        plt.xlabel(r'$\mathit{q}$',fontsize=20,color='k')
        plt.ylabel(r'$C_1\frac{q^\alpha-q}{\alpha-1}-(q-1)D$',fontsize=20,color='k')
        plt.title('Evaluation of   '+ r'$\mathit{q}_D$',fontsize=20,color='k')  
        ax=plt.gca()
        for xtick in ax.get_xticklabels():
            plt.setp(xtick,fontsize=14)
        for ytick in ax.get_yticklabels():
                plt.setp(ytick,fontsize=14)    
        plt.savefig('Determination_UM_par_3.png')
    

    return qD


############################################################################################
############################################################################################   

# This script allows to perform a spectral analysis. 
# The following options are available : 
#      - 1D or 2D analysis
#      - Sample analysis or average analysis on sets of samples
#      - Possibility to take into account up to three scaling regime
#      - Possibility to display or not the graphics


def spectral_analysis (data,data_file_name,k_range,dim,plot_index) :

    # Inputs :
    # - data : data is a numpy matrix that contain the data to be analysed. 
    #          Its meaning depends on dim
    #      - dim=1 : 1D analysis are performed
    #                data is a 2D matrix, each column is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                if one column, then a sample analysis is performed)
    #      - dim=2 : 2D analysis are performed
    #                data is a 3D matrix, each layer (i.e. data[:,:,s]) is a sample
    #                size(data)=[2^power,2^power,n], where n is number of samples
    #                (if one layer, then a sample analysis is performed)
    # - data_file_name : - if data_file_name =='' then "data" is used for the analysis
    #                    - otherwise : data_file_name is list of .csv file name
    #                                  the 1D or 2D field contained in each file is considered as a independant sample
    #                                  (used for large files)                
    # - k_range is a list of list. Each element contains two elements (k_min and k_max)
    #                and a spectral slope is evaluated between k_min and k_max
    #                if l_range=[] then a single scaling regime is considered and the all the available resolution are used
    #                the max length of k_range is 3; ie no more than three scaling regime can be studied
    # - plot_index : the number of the first figure opened for graph display 
    #
    # Outputs : 
    #  - bet_1 : the spectral slope of the scaling regime associated with l_range[0]
    #  - bet_2 : the spectral slope of the scaling regime associated with l_range[1] 
    #  - bet_3 : the spectral slope of the scaling regime associated with l_range[2]
    # Note 1 : the spectral slope "unused" scaling regime are returned as "nan"  

 
    # Evaluating l_max (maximum resolution) and nb_s (number of samples)
    if dim ==1:
        if data_file_name=='':
            l_max=data.shape[0]
            if len(data.shape)==1:
                nb_s=1
            else :    
                nb_s=data.shape[1]
        else:
            nb_s=len(data_file_name)
            l_max=np.loadtxt(data_file_name[0]).shape[0]
    elif dim==2:
        if data_file_name=='':
            l_max=data.shape[0]
            if len(data.shape)==2:
                nb_s=1
            else:
                nb_s=data.shape[2]
        else:
            nb_s=len(data_file_name)
            l_max=np.loadtxt(data_file_name[0],delimiter=';').shape[0]
    else:
        print('Error in spectral analysis, wrong dim')


    nb_k=np.floor(l_max/2)
    k=np.arange(1,nb_k+1,1)
    E=np.zeros((nb_k,))


    # Definition of the spectra
    if dim ==1 : 
        if data_file_name == '':
            if nb_s == 1: 
                buf=np.absolute(np.fft.fft(np.reshape(data,-1)))
                E[:]=buf[0:nb_k]*buf[0:nb_k]
            else :
                print('coucou 2')
                for s in range(nb_s) :
                    print(data[:,s].shape)                    
                    buf=np.absolute(np.fft.fft(data[:,s]))
                    E[:]=E[:]+buf[0:nb_k]*buf[0:nb_k]/nb_s
        else :
            for s in range(nb_s) :
                data_s=np.loadtxt(data_file_name[s],delimiter=';')
                buf=np.absolute(np.fft.fft(data_s))
                E[:]=E[:]+buf[0:nb_k]*buf[0:nb_k]/nb_s
                
    elif dim == 2 : 
        nb_el=np.zeros((nb_k,))
        if data_file_name == '':
            if nb_s == 1: 
                buf_1=np.absolute(np.fft.fft2(data))
                buf_2=buf_1[0:nb_k,0:nb_k]*buf_1[0:nb_k,0:nb_k]
                for i in range(sp.int0(nb_k)):
                    for j in range(sp.int0(nb_k)) : 
                        n=np.floor(np.sqrt((i+1)**2+(j+1)**2))
                        if (n-1)<nb_k : 
                            E[n-1]=E[n-1]+buf_2[i,j]
                            nb_el[n-1]=nb_el[n-1]+1
                for i in range(sp.int0(nb_k)):
                    E[i]=(i+1)*E[i]/nb_el[i]
            else :
                for s in range(nb_s) :
                    E_s=np.zeros((nb_k,))
                    buf_1=np.absolute(np.fft.fft2(data[:,:,s]))
                    buf_2=buf_1[0:nb_k,0:nb_k]*buf_1[0:nb_k,0:nb_k]
                    for i in range(sp.int0(nb_k)):
                        for j in range(sp.int0(nb_k)) : 
                            n=np.floor(np.sqrt((i+1)**2+(j+1)**2))
                            if (n-1)<nb_k : 
                                E[n-1]=E[n-1]+buf_2[i,j]
                                nb_el[n-1]=nb_el[n-1]+1
                    for i in range(sp.int0(nb_k)):
                        E[i]=(i+1)*E[i]/nb_el[i]
                    E=E+E_s/nb_s

        else :
            for s in range(nb_s) :
                E_s=np.zeros((nb_k,))
                data_s=np.loadtxt(data_file_name[s],delimiter=';')
                buf_1=np.absolute(np.fft.fft2(data_s))
                buf_2=buf_1[0:nb_k,0:nb_k]*buf_1[0:nb_k,0:nb_k]
                for i in range(sp.int0(nb_k)):
                    for j in range(sp.int0(nb_k)) : 
                        n=np.floor(np.sqrt((i+1)**2+(j+1)**2))
                        if (n-1)<nb_k : 
                            E[n-1]=E[n-1]+buf_2[i,j]
                            nb_el[n-1]=nb_el[n-1]+1
                for i in range(sp.int0(nb_k)):
                    E[i]=(i+1)*E[i]/nb_el[i]
                E=E+E_s/nb_s

    else: 
        print('Wrong dim in spectral_analysis')
    
    # Evaluation of the spectral slope according to the ranges defined in k_range
    # k_range is a list of list each containing 2 elt (k_min and k_max)
    # a linear regression is performed for each 
    y=sp.log(E)
    x=sp.log(k)
    if plot_index>0:
        plt.figure(plot_index)
        p1, = plt.plot(x,y,ls='None',marker='x',ms=9,mew=2,mfc='k',mec='k')     
        plt.xlabel(r'$\log(\mathit{k})$',fontsize=20,color='k')
        plt.ylabel(r'$\log(\mathit{E})$',fontsize=20,color='k')
        plt.title('Spectral analysis',fontsize=20,color='k')  
        ax=plt.gca()
        #plt.setp(ax.get_xticklabels(), fontsize=14) # it works but i test other things
        #plt.setp(ax.get_yticklabels(), fontsize=14)
        for xtick in ax.get_xticklabels():
            plt.setp(xtick,fontsize=14)
        for ytick in ax.get_yticklabels():
            plt.setp(ytick,fontsize=14)    


    nb_scale_reg = len(k_range) #Evaluation of the number of scaling regime    
    
    if nb_scale_reg == 0:
        # A single scaling regime, all the resolutions are considered
        a=sp.polyfit(x,y,1)   
        reg_lin=sp.poly1d(a)    
        bet_1=-a[0]
        r2=sp.corrcoef(x,y)[0,1]**2
        bet_2=np.nan
        bet_3=np.nan        
        if plot_index>0:
            p2, = plt.plot([x[0],x[-1]],[reg_lin(x[0]),reg_lin(x[-1])],lw=2,color='r')
            plt.legend([p2],[r'$\beta\ =\ $'+str(sp.floor(bet_1*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2*100)/100)],loc='upper right',frameon=False)
    elif nb_scale_reg == 1:
        # For the 1st scaling regime
        i_k_min=k_range[0][0]
        i_k_max=k_range[0][1]
        a=sp.polyfit(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1],1)  
        reg_lin=sp.poly1d(a)        
        bet_1=-a[0]
        bet_2=np.nan
        bet_3=np.nan
        if plot_index>0:
            r2=sp.corrcoef(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1])[0,1]**2
            p2, = plt.plot([x[i_k_min],x[i_k_max]],[reg_lin(x[i_k_min]),reg_lin(x[i_k_max])],lw=2,color='r')
            plt.legend([p2],[r'$\beta\ =\ $'+str(sp.floor(bet_1*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2*100)/100)],loc='upper right',frameon=False)
    elif nb_scale_reg == 2:
        # For the 1st scaling regime        
        i_k_min=k_range[0][0]
        i_k_max=k_range[0][1]
        a=sp.polyfit(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1],1)  
        reg_lin=sp.poly1d(a)   
        if plot_index>0:
            r2_1=sp.corrcoef(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1])[0,1]**2
            p2, = plt.plot([x[i_k_min],x[i_k_max]],[reg_lin(x[i_k_min]),reg_lin(x[i_k_max])],lw=2,color='b')
        bet_1=-a[0]
        # For the 2nd scaling regime        
        i_k_min=k_range[1][0]
        i_k_max=k_range[1][1]
        a=sp.polyfit(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1],1)  
        reg_lin=sp.poly1d(a) 
        bet_2=-a[0]
        bet_3=np.nan
        if plot_index>0:
            r2_2=sp.corrcoef(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1])[0,1]**2
            p3, = plt.plot([x[i_k_min],x[i_k_max]],[reg_lin(x[i_k_min]),reg_lin(x[i_k_max])],lw=2,color='r')
            plt.legend([p2,p3],[r'$\beta\ =\ $'+str(sp.floor(bet_1*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1*100)/100),r'$\beta\ =\ $'+str(sp.floor(bet_2*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_2*100)/100)],loc='upper right',frameon=False)         
    elif nb_scale_reg == 3:
        # For the 1st scaling regime        
        i_k_min=k_range[0][0]
        i_k_max=k_range[0][1]
        a=sp.polyfit(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1],1)  
        reg_lin=sp.poly1d(a)        
        if plot_index>0:
            r2_1=sp.corrcoef(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1])[0,1]**2
            p2, = plt.plot([x[i_k_min],x[i_k_max]],[reg_lin(x[i_k_min]),reg_lin(x[i_k_max])],lw=2,color='b')
        bet_1=-a[0]
        # For the 2nd scaling regime        
        i_k_min=k_range[1][0]
        i_k_max=k_range[1][1]
        a=sp.polyfit(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1],1)  
        reg_lin=sp.poly1d(a)        
        if plot_index>0:
            r2_2=sp.corrcoef(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1])[0,1]**2
            p3, = plt.plot([x[i_k_min],x[i_k_max]],[reg_lin(x[i_k_min]),reg_lin(x[i_k_max])],lw=2,color='r')
        bet_2=-a[0]
        # For the 3rd scaling regime        
        i_k_min=k_range[2][0]
        i_k_max=k_range[2][1]
        a=sp.polyfit(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1],1)  
        reg_lin=sp.poly1d(a)   
        bet_3=-a[0]
        if plot_index>0:
            r2_3=sp.corrcoef(x[i_k_min:i_k_max+1],y[i_k_min:i_k_max+1])[0,1]**2
            p4, = plt.plot([x[i_k_min],x[i_k_max]],[reg_lin(x[i_k_min]),reg_lin(x[i_k_max])],lw=2,color='g')
            plt.legend([p2,p3,p4],[r'$\beta\ =\ $'+str(sp.floor(bet_1*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_1*100)/100),r'$\beta\ =\ $'+str(sp.floor(bet_2*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_2*100)/100),r'$\beta\ =\ $'+str(sp.floor(bet_3*100)/100)+' ;  '+r'$\mathit{r}^2\ =\ $'+str(sp.floor(r2_3*100)/100)],loc='upper right',frameon=False)
    else :
        print('Error in spectral analysis, k_range has wrong size')
    
    plt.savefig('Spectral_analysis.png')
    
    return bet_1,bet_2,bet_3
    
        


############################################################################################
############################################################################################   
#
# The following functions enable to perform numerical simulations
# (The meaning being rather straighforward, no comments were added)
#


def beta_cascade(n,l,c,dim):

  from scipy.stats import uniform  
    
  if dim==1:
      data=np.ones(1)      
      for t in range(n):
         taille=sp.power(l,t+1)
         data_buf=np.zeros((taille,1))             
         for i in range(0,taille):
            i_prev=sp.floor((i+2)/l)-1
            u=uniform.rvs()
            if u<=sp.power(l,-c):
                data_buf[i]=data[i_prev]*sp.power(l,c)
            else:
                data_buf[i]=0
         data=data_buf           
  elif dim==2:
      data=np.ones((1,1))      
      for t in range(n):
         taille=sp.power(l,t+1)
         data_buf=np.zeros((taille,taille))             
         for i in range(0,taille):
             for j in range(0,taille):
                i_prev=sp.floor((i+2)/l)-1;
                j_prev=sp.floor((j+2)/l)-1;
                u=uniform.rvs();
                if u<=sp.power(l,-c):
                    data_buf[i,j]=data[i_prev,j_prev]*sp.power(l,c)
                else:
                    data_buf[i,j]=0
         data=data_buf  

  else:
    print('Wrong choice of dim in beta_cascade')

  return data

############################################################################################
############################################################################################

def levy(alpha):
    
    from scipy.stats import uniform, expon
    phi=uniform.rvs(loc=-sp.pi/2,scale=sp.pi)
    W=expon.rvs()
    
    if alpha!=1:
        phi0=(sp.pi/2)*(1-np.abs(1-alpha))/alpha;
        L=sp.sign(1-alpha)*(sp.sin(alpha*(phi-phi0)))*(((sp.cos(phi-alpha*(phi-phi0)))/W)**((1-alpha)/alpha))/((sp.cos(phi))**(1/alpha));
    else:
        print('Error : alpha = '+str(alpha)+' in Levy')
        
    return L



############################################################################################
############################################################################################


def cascade_discrete(n,l,alpha,C1,dim):
  data=np.ones((1,1))
  if dim==1:
      for t in range(n):       
         taille=sp.power(l,t+1)
         data_buf=np.zeros((taille,1))             
         for i in range(0,taille):
             i_prev=sp.floor((i+2)/l)-1
             L=levy(alpha)
             data_buf[i]=data[i_prev]*sp.exp(L*(C1*sp.log(l)/abs(alpha-1))**(1/alpha))/(l**(C1/(alpha-1)))  
         data=data_buf   
  elif dim==2:
      for t in range(n):        
         taille=sp.power(l,t+1)       
         data_buf=np.zeros((taille,taille))             
         for i in range(0,taille):
             for j in range(0,taille):
                 i_prev=sp.floor((i+2)/l)-1
                 j_prev=sp.floor((j+2)/l)-1
                 L=levy(alpha)
                 data_buf[i,j]=data[i_prev,j_prev]*sp.exp(L*(C1*sp.log(l)/abs(alpha-1))**(1/alpha))/(l**(C1/(alpha-1)))  
         data=data_buf 
  else :
        print('Wrong choice of dim in cascade_discrete')
   
   
  return data


############################################################################################
############################################################################################


def cascade_discrete_3D(n,l_xy,l_t,alpha,C1):
    data=np.ones((1,1,1))
    for t in range(n): 
        taille_xy=sp.power(l_xy,t+1)
        taille_t=sp.power(l_t,t+1)
        data_buf=np.zeros((taille_xy,taille_xy,taille_t))             
        for i in range(0,taille_xy):
            for j in range(0,taille_xy) :
                for k in range(0,taille_t):
                    i_prev=sp.floor((i+2)/l_xy)-1
                    j_prev=sp.floor((j+2)/l_xy)-1
                    k_prev=sp.floor((k+2)/l_t)-1
                    L=levy(alpha)
                    data_buf[i,j,k]=data[i_prev,j_prev,k_prev]*sp.exp(L*(C1*sp.log(l_xy)/abs(alpha-1))**(1/alpha))/(l_xy**(C1/(alpha-1)))  
                    
        data=data_buf   
   
    return data


############################################################################################
############################################################################################


def cascade_discrete_break(n,n_break,l,alpha1,C11,alpha2,C12,dim):
  data=np.ones((1,1))
  if dim==1:
      for t in range(n):
          if (t+1) <= n_break :
              alpha=alpha1
              C1=C11
          else :
              alpha=alpha2
              C1=C12
          taille=sp.power(l,t+1)
          data_buf=np.zeros((taille,1))             
          for i in range(0,taille):
              i_prev=sp.floor((i+2)/l)-1
              L=levy(alpha)
              data_buf[i]=data[i_prev]*sp.exp(L*(C1*sp.log(l)/abs(alpha-1))**(1/alpha))/(l**(C1/(alpha-1)))  
          data=data_buf   
  elif dim==2:
      for t in range(n):
          taille=sp.power(l,t+1)       
          data_buf=np.zeros((taille,taille))             
          for i in range(0,taille):
              for j in range(0,taille):
                  i_prev=sp.floor((i+2)/l)-1
                  j_prev=sp.floor((j+2)/l)-1
                  L=levy(alpha)
                  data_buf[i,j]=data[i_prev,j_prev]*sp.exp(L*(C1*sp.log(l)/abs(alpha-1))**(1/alpha))/(l**(C1/(alpha-1)))  
          data=data_buf 
  else :
        print('Wrong choice of dim in cascade_discrete_break')
   
   
  return data



############################################################################################
############################################################################################
#
# This functions computes the fluctuations of the field (in 1D for vectors or in 2D for matrix)
# See Lavallée et al. 1993 for more details
#
def fluctuations (data,dim) : 
    if dim == 1 :    
       i_max=data.shape[0]-1
       fluct_data=np.zeros((i_max+1,))       
       for i in range(i_max+1) :
           if i == 0 :
               fluct_data[i]=np.sqrt((data[i+1]-data[i])**2)
           elif (i>0) and (i<i_max) :
               fluct_data[i]=np.sqrt((data[i+1]-data[i-1])**2)
           elif i == i_max :
             fluct_data[i]=np.sqrt((data[i]-data[i-1])**2)
              
    elif dim == 2  :
        i_max=data.shape[0]-1 
        j_max=data.shape[1]-1
        fluct_data=np.zeros((i_max+1,j_max+1))    
        for i in range(i_max+1) :
            for j in range(j_max+1) : 
                if i==0 :
                    if j==0 :
                        fluct_data[i,j]=np.sqrt((data[i+1,j]-data[i,j])**2+(data[i,j+1]-data[i,j])**2)
                    elif (j>0) and (j<j_max) :
                        fluct_data[i,j]=np.sqrt((data[i+1,j]-data[i,j])**2+(data[i,j+1]-data[i,j-1])**2)
                    elif j==j_max :
                        fluct_data[i,j]=np.sqrt((data[i+1,j]-data[i,j])**2+(data[i,j]-data[i,j-1])**2)
                 
                elif (i>0) and (i<i_max):
                    if j==0 :
                        fluct_data[i,j]=np.sqrt((data[i+1,j]-data[i-1,j])**2+(data[i,j+1]-data[i,j])**2)
                    elif (j>0) and (j<j_max):
                        fluct_data[i,j]=np.sqrt((data[i+1,j]-data[i-1,j])**2+(data[i,j+1]-data[i,j-1])**2)
                    elif j==j_max :
                        fluct_data[i,j]=np.sqrt((data[i+1,j]-data[i-1,j])**2+(data[i,j]-data[i,j-1])**2)

                elif i==i_max:
                    if j==0 :
                        fluct_data[i,j]=np.sqrt((data[i,j]-data[i-1,j])**2+(data[i,j+1]-data[i,j])**2)
                    elif (j>0) and (j<j_max) :
                        fluct_data[i,j]=np.sqrt((data[i,j]-data[i-1,j])**2+(data[i,j+1]-data[i,j-1])**2)
                    elif j==j_max:
                        fluct_data[i,j]=np.sqrt((data[i,j]-data[i-1,j])**2+(data[i,j]-data[i,j-1])**2)
          
    return fluct_data






############################################################################################
############################################################################################
#
# This aim of this function is to show how to use the main functionalities 
# of the previous tools and check whether they are working.
#

def test_multifractal_tools ():
      
    print('Beginning of Test Multifractal Tools')
    
    print('Test 1D')
    
    print('Test with 1 sample and no scaling break')
    print('Pour sim : alpha=1.5 ; C1=0.1')
    n=12;
    l=2;
    alpha=1.5;
    C1=0.1;
    dim=1;
    
    data = cascade_discrete(n,l,alpha,C1,dim)
    
    data=data/np.mean(data)
    data2=np.zeros(data.shape)
    print('Threshold = 0.1')
    for i in range(data.shape[0]) :
        if data[i]<0.1 :
            data2[i]=0
        else :
            data2[i]=data[i]

    data_file_name=''
    dim=1
    l_range=[]
    file_index=0
    file_name='DF_test_truc.npy'
    plot_index=1
    D1, D2, D3 = fractal_dimension(data2,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('Fractal dimension')
    print('D1,D2,D3', D1,D2,D3)
    
    q_values=np.array([-1])
    file_name='TM_test_truc.npy'
    plot_index=5

    Kq_1,Kq_2,Kq_3,r2_1,r2_2,r2_3  = TM(data,q_values,data_file_name,dim,l_range,file_index,file_name,plot_index)

# TO BE DONE !!!!!!!!!!!!!!!!!!!!!!!    
#    print('TM : alpha and C1')
#    print([alpha1,C11])

    
    DTM_index=1;
    q_values=np.array([1.5])
    plot_index=10
    UM_par_1,UM_par_2,UM_par_3  = DTM(data,q_values,data_file_name,dim,l_range,DTM_index,file_index,file_name,plot_index)

    q_values=np.concatenate((np.arange(0.05,1,0.05),np.array([1.01]),np.array([1.05]),np.arange(1.1,3.1,0.1)),axis=0)
    alpha1=UM_par_1[0]
    C11=UM_par_1[1]
    H=0
    
    plt.figure(9)
    plt.plot(q_values,Kq_1,color='k',label='Empirical')
    Kq_theo=K_q (q_values,alpha,C1,H)
    plt.plot(q_values,Kq_theo,color='r',label='Theoretical with DTM UM par estimates')
    plt.legend(loc='upper right',frameon=False)
    
    print('DTM : alpha and C1')
    print([alpha1,C11])
    
    plot_index=15
    k_range=[]    
    plot_index=16
    bet1,bet2,bet3 = spectral_analysis (data,data_file_name,k_range,dim,plot_index)
    print('Spectral analysis')
    print('bet1,bet2,bet3',bet1,bet2,bet3)    
    
    
    print('Test with several samples and a scaling break')
    print('For sim : alpha1=0.9 ; C11=0.5; alpha2=1.7 ; C12=0.1')
    n=12
    n_break=7
    l=2
    alpha1=0.9
    C11=0.5
    alpha2=1.7
    C12=0.1
    dim=1
    
    data=np.zeros((4096,3))
    for s in range(0,3): 
        #data[:,s]=np.reshape(cascade_discrete(n,l,alpha,C1,dim) ,-1)
        data[:,s]=np.reshape(cascade_discrete_break(n,n_break,l,alpha1,C11,alpha2,C12,dim) ,-1)
        np.savetxt('data'+str(s)+'.csv',data[:,s],delimiter=';')
    
    data=data/np.mean(data);
#    //print(size(data))
#    print('Threshold = 0.1')
#    for s in range(3):
#        for i in range(data.shape[0]):
#            if data[i,s]<0.1:
#                data[i,s]=0

    l_range=[[1,128],[128,4096]]

    plot_index=20
    D1, D2, D3 = fractal_dimension(data,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('Fractal dimension')
    print('D1,D2,D3', D1,D2,D3)
    
    q_values=np.array([-1])
    file_name='TM_test_truc.npy'
    plot_index=25

    Kq_1,Kq_2,Kq_3,r2_1,r2_2,r2_3  = TM(data,q_values,data_file_name,dim,l_range,file_index,file_name,plot_index)

# TO BE DONE !!!!!!!!!!!!!!!!!!!!!!!    
#    print('TM : alpha and C1')
#    print([alpha1,C11])

    
    DTM_index=1;
    q_values=np.array([1.5])
    plot_index=30
    UM_par_1,UM_par_2,UM_par_3  = DTM(data,q_values,data_file_name,dim,l_range,DTM_index,file_index,file_name,plot_index)
    
    print('DTM alpha1 et C11')
    print(UM_par_1[0],UM_par_1[1])
    print('DTM alpha2 et C12')
    print(UM_par_2[0],UM_par_2[1])

    
    plot_index=35
    k_range=[[0,200],[200,2047]]
    bet1,bet2,bet3 = spectral_analysis (data,data_file_name,k_range,dim,plot_index)
    print('Spectral analysis')
    print('bet1,bet2,bet3',bet1,bet2,bet3)



   
    
    print('Test 2D')
    
    print('Test with 1 sample and no scaling break')
    print('Pour sim : alpha=1.5 ; C1=0.1')
    n=7;
    l=2;
    alpha=1.5;
    C1=0.1;
    dim=2;
    
    data=cascade_discrete(n,l,alpha,C1,dim)

    print('Threshold = 0.1')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j]<0.1:
                data[i,j]=0

    data_file_name=''
    dim=2
    l_range=[]
    file_index=0
    file_name='DF_test_truc.npy'
    plot_index=40
    D1, D2, D3 = fractal_dimension(data,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('Fractal dimension')
    print('D1,D2,D3', D1,D2,D3)
    
    q_values=np.array([-1])
    file_name='TM_test_truc.npy'
    plot_index=45

    Kq_1,Kq_2,Kq_3,r2_1,r2_2,r2_3  = TM(data,q_values,data_file_name,dim,l_range,file_index,file_name,plot_index)

# TO BE DONE !!!!!!!!!!!!!!!!!!!!!!!    
#    print('TM : alpha and C1')
#    print([alpha1,C11])

    
    DTM_index=1;
    q_values=np.array([1.5])
    plot_index=50
    UM_par_1,UM_par_2,UM_par_3  = DTM(data,q_values,data_file_name,dim,l_range,DTM_index,file_index,file_name,plot_index)

    q_values=np.concatenate((np.arange(0.05,1,0.05),np.array([1.01]),np.array([1.05]),np.arange(1.1,3.1,0.1)),axis=0)
    alpha1=UM_par_1[0]
    C11=UM_par_1[1]
    H=0
    
    plt.figure(9)
    plt.plot(q_values,Kq_1,color='k',label='Empirical')
    Kq_theo=K_q (q_values,alpha,C1,H)
    plt.plot(q_values,Kq_theo,color='r',label='Theoretical with DTM UM par estimates')
    plt.legend(loc='upper right',frameon=False)
    
    print('DTM : alpha and C1')
    print([alpha1,C11])
    
    plot_index=55
    k_range=[]    
    bet1,bet2,bet3 = spectral_analysis (data,data_file_name,k_range,dim,plot_index)
    print('Spectral analysis')
    print('bet1,bet2,bet3',bet1,bet2,bet3)    


    
    print('Test with several samples and no scaling break, input a 3D matrix')
    print('For sim : alpha=1.5 ; C1=0.1')
    n=7
    l=2
    alpha=1.5
    C1=0.1
    dim=2
    data=np.zeros((128,128,3))
    for s in range(0,3): 
        data[:,:,s]=cascade_discrete(n,l,alpha,C1,dim)
        np.savetxt('data2D'+str(s)+'.csv',data[:,:,s],delimiter=';')
    
    data_file_name=''
    dim=2
    l_range=[[1,128]]
    #l_range=[[1,256],[256,1024],[1024,4096]]
    file_index=0
    file_name='DF_test.npy'
    plot_index=60
        
    D1, D2, D3 = fractal_dimension(data,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('Fractal dimension')
    print('D1,D2,D3', D1,D2,D3)

    file_index=1
    D1, D2, D3 = fractal_dimension(data,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('Fractal dimension again with file_index=1')
    print('D1,D2,D3', D1,D2,D3)

    file_index=0
    q_values=np.array([-1])
    file_name='TM_test.npy'
    plot_index=65
    Kq_1,Kq_2,Kq_3,r2_1,r2_2,r2_3  = TM(data,q_values,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('TM : K_q')
    print(Kq_1)
    file_index=1    
    plot_index=70
    Kq_1,Kq_2,Kq_3,r2_1,r2_2,r2_3  = TM(data,q_values,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('TM : K_q again with file_index=1')
    print(Kq_1)

    
    DTM_index=1
    plot_index=75
    file_index=0
    q_values=np.array([1.5])
    file_name='DTM_test.npy'
    UM_par_1,UM_par_2,UM_par_3  = DTM(data,q_values,data_file_name,dim,l_range,DTM_index,file_index,file_name,plot_index)
    print('DTM : alpha and C1')
    print(UM_par_1[0],UM_par_1[1])

    file_index=1
    plot_index=80
    UM_par_1,UM_par_2,UM_par_3  = DTM(data,q_values,data_file_name,dim,l_range,DTM_index,file_index,file_name,plot_index)
    print('DTM : alpha and C1 again with file_index=1')
    print(UM_par_1[0],UM_par_1[1])

    DTM_index=2
    file_index=0
    plot_index=85
    UM_par_1,UM_par_2,UM_par_3  = DTM(data,q_values,data_file_name,dim,l_range,DTM_index,file_index,file_name,plot_index)
    print('DTM : alpha and C1 again with DTM_index=2')
    print(UM_par_1[0],UM_par_1[1])

    k_range=[]
    plot_index=90
    bet1,bet2,bet3 = spectral_analysis (data,data_file_name,k_range,dim,plot_index)
    print('Spectral analysis, bet1,bet2,bet3')
    print(bet1,bet2,bet3)

    
    print('Test with several samples and no scaling break, input with data_file_name')
    print('Results should be the same as in the previous test')

    data=np.array([-1])
    data_file_name=['data2D0.csv','data2D1.csv','data2D2.csv']

    file_name='DF_test.npy'
    plot_index=95
        
    D1, D2, D3 = fractal_dimension(data,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('Fractal dimension')
    print('D1,D2,D3', D1,D2,D3)

    file_index=0
    q_values=np.array([-1])
    file_name='TM_test.npy'
    plot_index=100
    Kq_1,Kq_2,Kq_3,r2_1,r2_2,r2_3  = TM(data,q_values,data_file_name,dim,l_range,file_index,file_name,plot_index)
    print('TM : K_q')
    print(Kq_1)
    
    DTM_index=1
    plot_index=105
    q_values=np.array([1.5])
    file_name='DTM_test.npy'
    UM_par_1,UM_par_2,UM_par_3  = DTM(data,q_values,data_file_name,dim,l_range,DTM_index,file_index,file_name,plot_index)
    print('DTM : alpha and C1')
    print(UM_par_1[0],UM_par_1[1])

    k_range=[]
    plot_index=110
    bet1,bet2,bet3 = spectral_analysis (data,data_file_name,k_range,dim,plot_index)
    print('Spectral analysis, bet1,bet2,bet3')
    print(bet1,bet2,bet3)    

    print('Hope you had fun performing multifractal analysis !')
    


         