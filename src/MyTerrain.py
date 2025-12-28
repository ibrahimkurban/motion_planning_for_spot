# python libraries
import numpy as np
from matplotlib.patches import Rectangle

from matplotlib import pyplot as plt
from pydrake.all import RigidTransform
from utils.utils_plot import plot_rectangle



#  stepping stone
class SteppingStone(object):
    def __init__(self, center, width, height, name=None):
        # store arguments
        self.center = center
        self.width = width
        self.height = height
        self.name = name
        self.V = []

        # distance from center to corners
        c2tr = np.array([width, height]) / 2
        c2br = np.array([width, -height]) / 2

        # position of the corners
        self.top_right = center + c2tr
        self.bottom_right = center + c2br
        self.top_left = center - c2br
        self.bottom_left = center - c2tr
        self.corners = [self.top_right, self.bottom_right, self.top_left, self.bottom_left]
        self.rear_end_x = self.center[0] - self.width/2
        self.front_end_x = self.center[0] + self.width/2
        self.top_end_y = self.center[1] + self.height/2
        self.down_end_y = self.center[1] - self.height/2


        # halfspace representation of the stepping stone
        self.A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.b = np.concatenate([c2tr] * 2) + self.A.dot(center)

    def plot(self, **kwargs):
        return plot_rectangle(self.center, self.width, self.height, **kwargs)
    
    def min_dist(self, other_stone):
        min_dist = np.inf
        for self_corner in self.corners:
                for other_corner in other_stone.corners:
                    dist = np.linalg.norm(self_corner - other_corner)
                    if dist < min_dist:
                        min_dist = dist
        return min_dist
    

# terrain object that uses stepping stones
class Terrain(object):
    # parametric construction of the stepping stones
    def __init__(
            self,
            init_center_xy,
            target_center_xy,
            init_wh,
            target_wh,
            num_bridges_y,
            num_stones_in_bridge,
            bridge_stone_wh,
            bridges_dist_y,
            bridges_centerline_y=0,
            bridge_x_shift=0,
            rand_radius=0,#should be in [0,1] for no overlapped stones
            rand_seed=42,
        ):

        self.init_center_xy = init_center_xy
        self.target_center_xy = target_center_xy
        self.init_wh = init_wh
        self.target_wh = target_wh
        self.num_bridges_y = num_bridges_y
        self.num_stones_in_bridge = num_stones_in_bridge
        self.bridge_stone_wh = bridge_stone_wh
        self.bridges_dist_y = bridges_dist_y
        self.bridges_centerline_y = bridges_centerline_y
        self.bridge_x_shift = bridge_x_shift
        self.rand_radius = rand_radius
        self.rand_seed = rand_seed


        # random seed
        rng = np.random.default_rng(seed=rand_seed)

        # initialize internal list of stepping stones
        self.stepping_stones = []

        # add initial stepping stone to the terrain
        self.initial = self.add_stone(init_center_xy, 2*init_wh[0], 2*init_wh[1], "initial")

        # horizontal gap between tones
        x_len = -init_center_xy[0] - init_wh[0] + target_center_xy[0] - target_wh[0]

        # number of bridge stones in horizontal axis
        if num_stones_in_bridge * (2*bridge_stone_wh[0]) < x_len:
            bridges_dist_x = x_len / (num_stones_in_bridge) 
        else:
            num_stones_in_bridge = int(x_len // (2*bridge_stone_wh[0]))
            # bridge width is too small
            if num_stones_in_bridge == 0:
                num_bridges_y = 0
                print(f'No bridge stone with half-width={bridge_stone_wh[0]} can be placed')
                print('Number of bridges is set to 0')
            else:
                bridges_dist_x = x_len / (num_stones_in_bridge)
                print(f"To accommodate desired bridge half-width, number of stones in one bridge is changed to {num_stones_in_bridge}")
        
        self.num_bridges_y = num_bridges_y
        if num_bridges_y > 0:
            # determine horizontal positions
            bridge_pos_x = np.linspace(
                -(num_stones_in_bridge-1)/2 * bridges_dist_x, 
                (num_stones_in_bridge-1)/2 * bridges_dist_x, 
                num_stones_in_bridge
            ) + (init_center_xy[0] + init_wh[0] + target_center_xy[0] - target_wh[0])/2 + bridge_x_shift

            # determine bridge_y's based on centerline and dist_y
            bridge_pos_y = np.linspace(
                -(num_bridges_y-1)/2 * bridges_dist_y, 
                (num_bridges_y-1)/2 * bridges_dist_y, 
                num_bridges_y
            ) + bridges_centerline_y

        # add bridges row by row
        # could be optimzied but no need for small scale
        for row in range(num_bridges_y):
            centers = np.zeros((num_stones_in_bridge,2))
            for col in range(num_stones_in_bridge):
                center = np.array([bridge_pos_x[col], bridge_pos_y[row]])
                rand_perturbation = np.hstack([
                    rng.uniform(-1, 1)*rand_radius*bridges_dist_x/2, 
                    rng.uniform(-1, 1)*rand_radius*bridges_dist_y/2, 
                ])
                centers[col] = center + rand_perturbation

            # add bridges:
            self.add_stones(
                    centers, 
                    [2*bridge_stone_wh[0]]*num_stones_in_bridge, 
                    [2*bridge_stone_wh[1]]*num_stones_in_bridge,
                    name=f'bridge{row}'
                )
            
        # add goal stepping stone to the terrain
        self.target = self.add_stone(target_center_xy, 2*target_wh[0], 2*target_wh[1], "target")

        if num_bridges_y:
            self.num_stones_in_bridge = num_stones_in_bridge
            self.bridges_dist_x = bridges_dist_x
            self.bridges_dist_y = bridges_dist_y


    # adds a stone to the internal list stepping_stones
    def add_stone(self, center, width, height, name=None):
        stone = SteppingStone(center, width, height, name=name)
        self.stepping_stones.append(stone)
        return stone

    # adds multiple stones to the internal list stepping_stones
    def add_stones(self, centers, widths, heights, name=None):
        # ensure that inputs have coherent size
        n_stones = len(centers)
        if n_stones != len(widths) or n_stones != len(heights):
            raise ValueError("Arguments have incoherent size.")

        # add one stone per time
        stones = []
        for i in range(n_stones):
            stone_name = name if name is None else name + "_" + str(i)
            stones.append(
                self.add_stone(centers[i], widths[i], heights[i], name=stone_name)
            )

        return stones

    def erase_bridge_stone(self, indices):
        for ind in sorted(indices, reverse=True):
            del self.stepping_stones[ind]
            
    # returns the stone with the given name
    # raise a ValueError if no stone has the given name
    def get_stone_by_name(self, name):
        # loop through the stones
        # select the first with the given name
        for stone in self.stepping_stones:
            if stone.name == name:
                return stone

        # raise error if there is no stone with the given name
        raise ValueError(f"No stone in the terrain has name {name}.")

    # plots all the stones in the terrain
    def plot(self, title=None, **kwargs):
        # make light green the default facecolor
        if not "facecolor" in kwargs:
            kwargs["facecolor"] = [0, 1, 0, 0.1]

        # plot stepping stones disposition
        labels = ["Stepping stone", None]
        for i, stone in enumerate(self.stepping_stones):
            stone.plot(label=labels[min(i, 1)], **kwargs)
            
            # print Index ---
            # Calculate the top-right corner coordinates
            tr_x = stone.center[0] + stone.width / 2.0
            tr_y = stone.center[1] + stone.height / 2.0

            # Add text at that coordinate
            # ha='left', va='bottom' aligns the text so it sits outside the top-right corner
            plt.text(tr_x, tr_y, str(i), 
                     fontsize=6, 
                     color='black', 
                     ha='left', 
                     va='bottom')

        # plt.ylim(-1 ,1)   
        # set title
        plt.title(title)