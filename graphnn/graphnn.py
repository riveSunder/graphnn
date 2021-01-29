import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot

def atom_tokens(my_seed=13):
    np.random.seed(my_seed)
    token_length = 4

    tokens = {}

    for my_key in "CHNOPS":
        tokens[my_key] = np.random.randn(token_length)


    return tokens

def parse_pdbqt(directory="data/ligands/", atom_dictionary=None):

    my_dir = os.listdir(directory)

    if directory[-1] is not "/":
        directory += "/"

    if atom_dictionary is None:
        atom_dictionary = atom_tokens()

    ligands = []
    raw_ligands = []

    for filename in my_dir:
        raw_nodes = []
        nodes = np.array([])
        temp = ""

        with open(directory + filename) as f:
            my_axis = None
            while "TORSDOF" not in temp:

                temp = f.readline()

                if "ATOM" in temp:
                    raw_nodes.append([filename])
                    raw_nodes[-1].extend(temp.split())

                
                    # add atom positions
                    my_node = np.array(raw_nodes[-1][6:9], dtype=np.float) 
                    
                    for my_key in atom_dictionary.keys():
                        if my_key in raw_nodes[-1][3]:
                            token = atom_dictionary[my_key]

                    my_node = np.append(my_node, token)

                    nodes = np.append(nodes,\
                            my_node.reshape(1, my_node.shape[0]), axis=my_axis)

                    my_axis = 0

                    if len(nodes.shape) < 2:
                        nodes = nodes[np.newaxis,:] 

            mean_x = np.mean(nodes[:,0])
            mean_y = np.mean(nodes[:,1])
            mean_z = np.mean(nodes[:,2])

            nodes[:,0] -= mean_x
            nodes[:,1] -= mean_y
            nodes[:,2] -= mean_z

        raw_ligands.append(raw_nodes)
        ligands.append(nodes)

    return ligands, raw_ligands

class ArcTan(nn.Module):

    def init(self):
        super(ArcTan,self).__init__()

    def forward(self, x):

        return torch.arctan(x)

class GraphNN(nn.Module):

    def init(self, ligand_dim=7):
        super(GraphNN, self).__init__()
        
        self.ligand_dim = ligand_dim
        self.dim_h = 16
        # This is a guesstimate based on: 
        # https://pymolwiki.org/index.php/Displaying_Biochemical_Properties
        self.bond_cutoff = 3.6

        self.initialize()
        self.reset()

    def initialize(self):

        # vertices MLP, with 8 element key and query vectors for self-attention
        self.model = nn.Sequential(\
                nn.Linear(self.ligand_dim, self.di_h),\
                ArcTan(),\
                nn.Linear(self.dim_h, self.dim_h),\
                ArcTan(),\
                nn.Conv2D(self.dim_h, self.ligand_dim + 8 + 8)
                )

        self.encoder = nn.Sequential(\
                nn.Linear(self.ligand_dim, self.dim_h),\
                ArcTan()\
                )

        self.decoder = nn.Sequential(\
                nn.Linear(self.dim_h, self.ligand_dim),\
                )
        
    def get_distance(self, node_0, node_1):

        return torch.sum(torch.sqrt(torch.abs(node_0 - node_1)**2))

    def build_graph(self, x):

        self.graph = torch.zeros(x.shape[0],x.shape[0])

        for ii in range(x.shape[0]):
            node_ii = x[ii, 0:3]
            for jj in range(x.shape[0]):
                node_jj = x[jj, 0:3]

                distance = self.get_distance(node_0, node_1)
                if distance <= self.bond_cutoff:
                    self.graph[ii, jj] = 1.0
                

    def forward(self, x):

        self.build_graph(x)

        for kk in range(x.shape[0]):
            # loop through nodes for each node
            for ll in range(x.shape[0]):

                pass

    def reset(self):
        # initialize using gated cell states here later (maybe)
        pass


def train_ligannd():

    pass

if __name__ == "__main__":

    directory = "data/ligands"

    atom_dictionary = atom_tokens()
    nodes, raw_nodes = parse_pdbqt(directory)
    import pdb; pdb.set_trace()


