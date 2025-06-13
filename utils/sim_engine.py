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

    def simulation_step(self, dt=0.02, k_attraction=0.05, k_repulsion=50.0, damping=0.95):
        """Performs one step of the physics simulation."""
        if self.num_nodes == 0:
            return

        forces = np.zeros((self.num_nodes, 2), dtype=np.float32)

        # 1. Calculate Geometric Centroids (Dynamic)
        for i, he in enumerate(self.hyperedges):
            if he:
                member_positions = self.positions[he]
                self.centroids[i] = np.mean(member_positions, axis=0)

        # 2. Calculate Node-Centroid Attraction Forces
        for i in range(self.num_nodes):
            for he_idx in self.node_to_hyperedges[i]:
                force_vec = self.centroids[he_idx] - self.positions[i]
                forces[i] += force_vec # k_attraction is applied later

        forces *= k_attraction

        # 3. Calculate Node-Node Repulsion (Approximated for performance)
        # A full N^2 is too slow. We use a random sample.
        # For a more robust solution, a Quadtree is needed.
        num_samples = 100 # Tune this for balance of performance/quality
        for i in range(self.num_nodes):
            # Select random samples, excluding self
            samples_idx = np.random.choice(self.num_nodes, num_samples, replace=False)
            
            delta = self.positions[samples_idx] - self.positions[i]
            distance_sq = np.sum(delta**2, axis=1)
            distance_sq[distance_sq == 0] = 1e-6 # Avoid division by zero
            
            # Simplified force calculation
            force_magnitude = k_repulsion / distance_sq
            repulsion_force = np.sum(delta * force_magnitude[:, np.newaxis], axis=0)
            forces[i] -= repulsion_force / num_samples # Average the force

        # 4. Update Physics (Euler Integration)
        self.velocities += forces * dt
        self.velocities *= damping
        self.positions += self.velocities * dt

