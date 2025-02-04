import pandas as pd
import torch
import numpy as np

def to_csv(folder_names, times) :
    ''' 
    Create and store the equivalent csv files for all data files stored in the folder ./folder_name

    folder_name = string of the folder name containing all data
    times = array containing the times on which the qantities have been computed
    '''
    
    for folder_name in folder_names :
        for i in range(1, len(times)+1):
            
            # Input and output file paths
            if i < 10 :
                i_path = '00'+str(i)
            elif i >= 10 and i < 100 :
                i_path = '0'+str(i)
            else :
                i_path = str(i)
                
            input_file = f"./{folder_name}/data-0{i_path}" 
            output_file = f"./{folder_name}/data-0{i_path}.csv" 

            df = pd.read_csv(input_file, delim_whitespace=True, skiprows=1, 
                            names=['nodenumber', 'x-coordinate', 'y-coordinate', 'y-velocity', 'x-velocity', 'pressure'])

            # Scientific notation from E to e so python can understand it as a number
            for col in ['x-coordinate', 'y-coordinate', 'y-velocity', 'x-velocity', 'pressure']:
                df[col] = df[col].apply(lambda x: f"{x:.10e}")

            # create the equivalent csv file
            df.to_csv(output_file, index=False)



def get_data_MLP(folder_name, times) :
    ''' 
    Argument :
        folder_name = string of the folder name containing all data
        times = array containing the times on which the qantities have been computed

    Return :
        X : torch array containing the inputs data associated to the folder
        Y : torch array containing the outputs data associated to the folder
    '''
    

    df_example = pd.read_csv(f"./{folder_name}/data-0001.csv")
    
    # same grid for each dt -> same number of nodes in each df
    nb_nodes = len(df_example)

    # Input = (x, y, t)
    X = torch.empty(len(times)*nb_nodes, 3)
    
    # Output = (p(x, y, t), ux(x, y, t), uy(x, y, t))
    Y = torch.empty(len(times)*nb_nodes, 3)

    for i in range(1, len(times)+1):
        
        t = times[i-1]
        
        if i < 10 :
            i_path = '00'+str(i)
        elif i >= 10 and i < 100 :
            i_path = '0'+str(i)
        else :
            i_path = str(i)

        df_t = pd.read_csv(f"./{folder_name}/data-0{i_path}.csv")

        x_coord = torch.from_numpy(df_t['x-coordinate'].values)
        y_coord = torch.from_numpy(df_t['y-coordinate'].values)
        pressure = torch.from_numpy(df_t['pressure'].values)
        x_velocity = torch.from_numpy(df_t['x-velocity'].values)
        y_velocity = torch.from_numpy(df_t['y-velocity'].values)

        start_idx = (i-1)*nb_nodes
        end_idx = i*nb_nodes

        X[start_idx:end_idx, 0] = x_coord
        X[start_idx:end_idx, 1] = y_coord
        X[start_idx:end_idx, 2] = t

        Y[start_idx:end_idx, 0] = pressure
        Y[start_idx:end_idx, 1] = x_velocity
        Y[start_idx:end_idx, 2] = y_velocity

    return X, Y

def merge_traindata(folder_names, times) :
    ''' 
    Argument :
        folder_name = array of strings of the folder name containing all data
        times = array containing the times on which the qantities have been computed

    Return :
        X : torch array containing the stacked inputs data associated to the all folders
        X : torch array containing the stacked outputs data associated to the all folders
    '''
    
    lengths = np.zeros(len(folder_names))
    for i in range(len(folder_names)) :
        folder_name = folder_names[i]
        nb_nodes = len(pd.read_csv(f"./{folder_name}/data-0001.csv"))
        lengths[i] = nb_nodes
    
    X, Y = get_data_MLP(folder_names[0], times)
    
    for i in range(1, len(folder_names)) :
        folder_name = folder_names[i]
        X_temp, Y_temp = get_data_MLP(folder_name, times)

        X = torch.stack([X, X_temp], dim=0)
        Y = torch.stack([Y, Y_temp], dim=0)
    
    return X, Y