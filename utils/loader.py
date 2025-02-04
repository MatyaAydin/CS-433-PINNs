
from torch.utils.data import  Dataset


class SteadyDataset(Dataset):
    """"class to train on batch of steady state data"""
    def __init__(self, X, X_left, X_right, X_bot, X_top, Y_left, Y_right, Y_bot, Y_top):
        self.X = X

        self.X_left = X_left
        self.X_right = X_right
        self.X_bot = X_bot
        self.X_top = X_top

        self.Y_left = Y_left
        self.Y_right = Y_right
        self.Y_bot = Y_bot
        self.Y_top = Y_top

    def __len__(self):
        #only batch on X
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
    

#prendre celui d'ibrahim
class UnsteadyDataset(Dataset):
    """"class to train on batch of unsteady state data"""
    def __init__(self, X, Y, X_grid, X_wall, Y_wall,):
        self.X = X
        self.Y = Y

        self.X_grid = X_grid
        self.X_wall = X_wall
        self.Y_wall = Y_wall


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.X_grid[idx], self.X_wall[idx], self.Y_wall[idx]
    
    

