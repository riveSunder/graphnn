import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import gc

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

    def __init__(self):
        super(ArcTan,self).__init__()

    def forward(self, x):

        return torch.arctan(x)

class GraphNN(nn.Module):

    def __init__(self, ligand_dim=7):
        super(GraphNN, self).__init__()
        
        self.ligand_dim = ligand_dim
        self.dim_h = 8
        # This is a guesstimate based on: 
        # https://pymolwiki.org/index.php/Displaying_Biochemical_Properties
        self.bond_cutoff = 3.6

        self.initialize_gnn()
        self.reset_state()

    def initialize_gnn(self):

        # vertices MLP, with 8 element key and query vectors for self-attention
        self.model = nn.Sequential(\
                nn.Linear(self.ligand_dim, self.dim_h),\
                ArcTan(),\
                nn.Linear(self.dim_h, self.dim_h),\
                ArcTan(),\
                nn.Linear(self.dim_h, self.ligand_dim + 8 + 8)
                )

        self.encoder = nn.Sequential(\
                nn.Linear(2*self.ligand_dim, self.dim_h),\
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

                distance = self.get_distance(node_ii, node_jj)
                if distance <= self.bond_cutoff:
                    self.graph[ii, jj] = 1.0
                

    def forward(self, x, return_codes=False, template=None):

        if template is not None:
            self.build_graph(template.detach())
        else:
            self.build_graph(x.detach())
        
        new_graph = torch.Tensor() #torch.zeros_like(x)
        codes = torch.Tensor() #torch.zeros(x.shape[0], self.dim_h)
        temp_input = [torch.Tensor()] 
        #orch.Tensor() #torch.zeros(x.shape[0], self.dim_h+8+8)

        for kk in range(x.shape[0]):
            # loop through nodes for each node
            for ll in range(x.shape[0]):
                if self.graph[kk,ll]:
                    temp_input[-1] = torch.cat([temp_input[-1],\
                            self.model(x[ll]).unsqueeze(0)])

            keys = temp_input[-1][:,-16:-8]
            queries = temp_input[-1][:,-8:]

            attention = torch.zeros(1, keys.shape[0])

            for mm in range(keys.shape[0]):
                attention[:, mm] = torch.matmul(queries[mm], keys[mm].T)

            attention = torch.softmax(attention, dim=1)

            my_input = torch.sum(attention.T \
                    * temp_input[-1][:,:self.ligand_dim],dim=0)
            my_input = torch.cat([x[kk], my_input])

            #this is where the cell gating would happen (TODO)
            codes = torch.cat([codes, self.encoder(my_input).unsqueeze(0)])

            new_graph = torch.cat([new_graph, self.decoder(codes[-1]).unsqueeze(0)])


        if return_codes:
            return codes, new_graph
        else:
            return new_graph


    def reset_state(self):
        # initialize using gated cell states here later (maybe)
        pass



if __name__ == "__main__":

    directory = "data/ligands"
    num_epochs = 5000
    num_steps = 8
    noise_scale = torch.Tensor([1e-2,1e-2,1e-2,0.0,0.0,0.0,0.0])
    learning_rate = 1e-4

    atom_dictionary = atom_tokens()
    nodes, raw_nodes = parse_pdbqt(directory)
    
    nodes = [torch.Tensor(elem) for elem in nodes]

    gnn = GraphNN() 
    optimizer = torch.optim.Adam(gnn.parameters(), lr=learning_rate)

    losses = []
    try:
        for epoch in range(num_epochs):


            for ligand in nodes:
                gnn.zero_grad()
                ligand_in = ligand.clone() \
                        + noise_scale * torch.randn_like(ligand) 
                for step in range(num_steps):
                    ligand_in = gnn(ligand_in, template=ligand)

                loss = torch.mean(torch.abs(ligand[0:3]-ligand_in[0:3])**2)

                loss.backward()
                optimizer.step()
            losses.append(loss.detach())
            print("loss at epoch {} = {:.3e}".format(epoch, loss))
            gc.collect()
    except KeyboardInterrupt:
        pass


    import pdb; pdb.set_trace()



            




            
