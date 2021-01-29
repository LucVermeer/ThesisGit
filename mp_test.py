import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from mpl_toolkits.mplot3d import Axes3D
from noisyopt import minimizeSPSA

class Model():
    def __init__(self, pred_die=0.02, pred_repr=0.5, prey_repr=0.9, step_size = 100, mean_pred=500, mean_prey=1000):
        # grid dimensions
        self.width = 75
        self.height = 75
        
        # the values of different states of cells (containing predator, prey or none) 
        self.empty_val = 0
        self.pred_val = 1
        self.prey_val = 2
        
        self.pred_die_probability = pred_die
        self.pred_reproduce_probability = pred_repr
        self.prey_reproduce_probability = prey_repr
        self.step_size = step_size

        self.coordinates = []
        for y in range(self.height):
            for x in range(self.width):
                self.coordinates.append((y, x))
                
        self.error = 0
        
        # define starting population
        n_pred_start = 1500
        n_prey_start = 2*n_pred_start
        
        try: 
            (n_pred_start+n_prey_start) < self.width*self.height
        except ValueError:
            print("Population exceeds the gridsize")
        
        self.mean_pred = n_pred_start
        self.mean_prey = n_prey_start
        
        self.n_pred = n_pred_start
        self.n_prey = n_prey_start
        
        # init grid
        grid_flat = np.zeros(self.width*self.height)
        grid_flat[0:n_pred_start] = self.pred_val
        grid_flat[n_pred_start:n_pred_start+n_prey_start] = self.prey_val
        np.random.shuffle(grid_flat)
        self.grid = np.reshape(grid_flat, (self.height, self.width))
        
    def select_random_neighbor(self, coordinates):
        y = (coordinates[0] + (np.random.randint(3) - 1)) % self.height
        x = (coordinates[1] + (np.random.randint(3) - 1)) % self.width
        return (y, x)

    def step_grid(self):
        np.random.shuffle(self.coordinates)
        for coordinate in self.coordinates:
            value = self.grid[coordinate]
            if value == self.empty_val:
                continue
            else:
                neighbor_coordinate = self.select_random_neighbor(coordinate)
                neighbor_val = self.grid[neighbor_coordinate]
                if value == self.pred_val:
                    # predator
                    if neighbor_val == self.empty_val:
                        # die maybe
                        self.grid[coordinate] = self.empty_val
                        self.n_pred -= 1
                        if np.random.rand() > self.pred_die_probability:
                            # move (do not die)
                            self.grid[neighbor_coordinate] = self.pred_val
                            self.n_pred += 1
                    if neighbor_val == self.prey_val:                        
                        self.n_prey -= 1
                        if np.random.rand() < self.pred_reproduce_probability:
                            # eat and reproduce in prey's cell
                            self.grid[neighbor_coordinate] = self.pred_val
                            self.n_pred += 1
                        else:
                            # only eat prey
                            self.grid[neighbor_coordinate] = self.empty_val
                elif value == self.prey_val:
                    # prey
                    if neighbor_val == self.empty_val:
                        # (reproduce to/) move to new cell
                        self.grid[neighbor_coordinate] = self.prey_val
                        self.n_prey += 1
                        if np.random.rand() > self.prey_reproduce_probability:
                            # (do not reproduce/ kill old cell)
                            self.grid[coordinate] = self.empty_val
                            self.n_prey -= 1
                    if neighbor_val == self.pred_val:
                        self.n_prey -= 1
                        if np.random.rand() < self.pred_reproduce_probability:
                            # eat and reproduce to prey's cell
                            self.grid[coordinate] = self.pred_val
                            self.n_pred += 1
                        else:
                            # only eat prey
                            self.grid[coordinate] = self.empty_val

                            
    def run_model(self):
        pred_population = []
        prey_population = []
        for i in trange(self.step_size):
            pred_population.append(self.n_pred)
            prey_population.append(self.n_prey)
            if self.n_prey == 0 or self.n_pred == 0:
                break
            self.step_grid()
        plt.plot(pred_population, label="predators")
        plt.plot(prey_population, label="prey")
        plt.legend()
        self.objective_function((pred_population, prey_population))
        return (pred_population, prey_population)
    
    def run_of(self):
        pred_population = []
        prey_population = []
        for i in trange(self.step_size):
            pred_population.append(self.n_pred)
            prey_population.append(self.n_prey)
            if self.n_prey == 0 or self.n_pred == 0:
                break
            self.step_grid()
        plt.plot(pred_population, label="predators")
        plt.plot(prey_population, label="prey")
        plt.legend()
        plt.draw()
        return self.objective_function((pred_population, prey_population))
    
#     def objective_function(self, time_series):
#         if len(time_series[0]) != self.step_size or len(time_series[1]) != self.step_size:
#             return 1000000000
#         mean_0 = np.mean(time_series[0])
#         mean_1 = np.mean(time_series[1])
#         cost = 0
#         for i in range(len(time_series[0])):
#             cost += (time_series[0][i] - mean_0)**2
#             cost += (time_series[1][i] - mean_1)**2
#         self.error = cost
#         return cost
    
    
    def objective_function(self, time_series):
        if len(time_series[0]) != self.step_size or len(time_series[1]) != self.step_size:
            return 5000
        pred_population = np.array(time_series[0][-150:])
        prey_population = np.array(time_series[1][-150:])
        cost = max(abs(prey_population-self.mean_prey)+abs(pred_population-self.mean_pred))
        self.error = cost
        return cost
        

import multiprocessing as mp

def f(x1=0.5, x2=0.5, x3=0.5):
    for i in trange(3):
        print(Model(x1, x2, x3, 300).run_of())



if __name__ == '__main__':
    q = mp.Queue()
    p = mp.Process(target=f, args=(0.5, 0.5, 0.5))
    print("starting process")
    p.start()
    print(q.get())
    p.join()
    exit()