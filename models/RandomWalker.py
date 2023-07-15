import numpy as np

def random_walk(adj, rwlen, n_walks):
    locations = np.arange(len(adj))
    walk_seq = [] #游走的seq
    for w in range(n_walks): #walk number= 100
        walk = []
        # walk_neg = []
        curr_node = np.random.choice(len(adj))
        walk.append(curr_node)
        for it in range(rwlen): #walk steps = 10
            pp = adj[curr_node,:]
            curr_node = np.random.choice(locations, size=1, p = pp.reshape(-1))[0]
            walk.append(curr_node)
        walk_seq.append(walk)
    return np.array(walk_seq)

class RW():
    """Helper class to generate random walks on the input adjacency matrix."""
    def __init__(self, adj, rw_len, n_walks):
        super(RW, self).__init__()
        self.adj = adj
        self.rw_len = rw_len
        self.n_walks = n_walks

    def walk(self):
        while True:
            yield random_walk(self.adj, self.rw_len, self.n_walks).reshape([-1, self.rw_len+1])