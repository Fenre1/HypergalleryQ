import numpy as np

class SimulationEngine:
    def __init__(self, initial_positions, hyperedges):
        self.num_nodes = len(initial_positions)
        self.positions = np.copy(initial_positions).astype(np.float32)
        self.velocities = np.zeros((self.num_nodes, 2), dtype=np.float32)

        # Build efficient lookups for hyperedge membership
        self.hyperedges = [list(he) for he in hyperedges] # Ensure lists for indexing
        self.num_hyperedges = len(self.hyperedges)
        
        self.node_to_hyperedges = [[] for _ in range(self.num_nodes)]
        for he_idx, he in enumerate(self.hyperedges):
            for node_idx in he:
                self.node_to_hyperedges[node_idx].append(he_idx)
                
        self.centroids = np.zeros((self.num_hyperedges, 2), dtype=np.float32)

        # Build once, inside __init__
        node_indices   = []
        hedge_indices  = []
        for h_id, nodes in enumerate(hyperedges):
            node_indices.extend(nodes)
            hedge_indices.extend([h_id] * len(nodes))

        self.mem_nodes  = np.asarray(node_indices,  dtype=np.int32)
        self.mem_hedges = np.asarray(hedge_indices, dtype=np.int32)

        # Constant factors
        self.hyper_cardinality = np.bincount(self.mem_hedges)[:, None].astype(np.float32)
        self.node_degrees = np.array([len(he_list) for he_list in self.node_to_hyperedges], dtype=np.float32)


    def simulation_step(self, dt=0.02, k_attraction=0.2,
                    k_repulsion=0.01, damping=0.95, num_samples=100):

        if self.num_nodes == 0:
            return

        forces = np.zeros_like(self.positions)

        # --- 1. centroids ---
        centroid_sum = np.zeros_like(self.centroids)
        np.add.at(centroid_sum, self.mem_hedges,
                self.positions[self.mem_nodes])
        self.centroids[:] = centroid_sum / self.hyper_cardinality

        # --- 2. attraction ---
        diff = (self.centroids[self.mem_hedges] -
                self.positions[self.mem_nodes]) * k_attraction
        np.add.at(forces, self.mem_nodes, diff)

        # --- 2.5. Inter-Centroid Attraction (NEW SECTION) ---
        # Warning: This is computationally expensive (O(num_hyperedges^2))
        k_centroid_attraction = 0.01 # A new parameter to tune
        all_dist = []
        for i in range(self.num_hyperedges):
            for j in range(i + 1, self.num_hyperedges):
                centroid_i = self.centroids[i]
                centroid_j = self.centroids[j]
                
                # Simple distance-based similarity
                dist = np.linalg.norm(centroid_i - centroid_j)
                all_dist.append(dist)
        t1 = np.max(all_dist)
        for i in range(self.num_hyperedges):
            for j in range(i + 1, self.num_hyperedges):
                centroid_i = self.centroids[i]
                centroid_j = self.centroids[j]
                
                # Simple distance-based similarity
                dist = np.linalg.norm(centroid_i - centroid_j)
                # Only attract if they are reasonably close
                if dist <= t1: 
                    # Force pulling centroid i towards j
                    force_vec = (centroid_j - centroid_i) * k_centroid_attraction
                    
                    # Apply this force to all members of hyperedge i
                    nodes_in_i = self.hyperedges[i]
                    forces[nodes_in_i] += force_vec

                    # Apply the opposite force to all members of hyperedge j
                    nodes_in_j = self.hyperedges[j]
                    forces[nodes_in_j] -= force_vec


        # --- 3. sampled repulsion (MODIFIED LOGIC) ---
        samples = np.random.randint(0, self.num_nodes,
                                    size=(self.num_nodes, num_samples),
                                    dtype=np.int32)
        delta   = self.positions[samples] - self.positions[:, None, :]
        dist2   = (delta**2).sum(-1, keepdims=True)
        dist2[dist2 == 0.0] = 1e-6

        # Make repulsion strength dependent on node degree
        # We reshape node_degrees to (num_nodes, 1, 1) to broadcast correctly
        per_node_k_repulsion = k_repulsion * self.node_degrees[:, None, None]

        forces -= (per_node_k_repulsion * delta / dist2).mean(axis=1)

    

