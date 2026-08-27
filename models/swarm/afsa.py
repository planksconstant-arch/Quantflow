"""
Artificial Fish Swarm Algorithm (AFSA) Optimizer
================================================
Continuous bionic optimization algorithm based on animal foraging and schooling
behaviors: Preying (Searching), Swarming (Clustering), Chasing (Following), and Leaping.

Used for offline and online calibration of LOB feature weights, Hawkes kernels,
and Avellaneda-Stoikov inventory aversion parameters.
"""

import numpy as np
from typing import Callable, Tuple, List, Dict


class ArtificialFishSwarmOptimizer:
    """
    Artificial Fish Swarm Algorithm (AFSA) for multi-parameter global optimization.
    """
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        pop_size: int = 25,
        visual_radius: float = 0.5,
        step_size: float = 0.1,
        crowding_factor: float = 0.618,
        try_number: int = 5,
        maximize: bool = True,
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.visual = visual_radius
        self.step_size = step_size
        self.delta = crowding_factor
        self.try_number = try_number
        self.maximize = maximize
        
        # Initialize swarm population within bounds
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.population = np.random.uniform(low, high, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        
        # Evaluate initial fitness
        for i in range(pop_size):
            self.fitness[i] = self._eval(self.population[i])
            
        # Global best tracker
        best_idx = np.argmax(self.fitness) if self.maximize else np.argmin(self.fitness)
        self.best_x = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.history = [self.best_fitness]

    def _eval(self, x: np.ndarray) -> float:
        """Evaluate objective function with boundary clamping"""
        clamped_x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        try:
            return self.objective_func(clamped_x)
        except Exception:
            return -1e9 if self.maximize else 1e9

    def _is_better(self, fit1: float, fit2: float) -> bool:
        return fit1 > fit2 if self.maximize else fit1 < fit2

    def prey(self, i: int) -> np.ndarray:
        """
        Preying / Searching behavior: Fish senses food density in random visual direction.
        """
        xi = self.population[i]
        fi = self.fitness[i]
        
        for _ in range(self.try_number):
            # Random point within visual field
            rand_dir = np.random.uniform(-1.0, 1.0, size=self.dim)
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-8:
                rand_dir = (rand_dir / norm) * np.random.uniform(0.0, self.visual)
            xj = xi + rand_dir
            xj = np.clip(xj, self.bounds[:, 0], self.bounds[:, 1])
            fj = self._eval(xj)
            
            if self._is_better(fj, fi):
                # Move toward xj
                direction = (xj - xi) / (np.linalg.norm(xj - xi) + 1e-8)
                step_move = xi + direction * self.step_size * np.random.uniform(0.5, 1.0)
                return np.clip(step_move, self.bounds[:, 0], self.bounds[:, 1])
                
        # Default random move
        rand_move = xi + np.random.uniform(-self.step_size, self.step_size, size=self.dim)
        return np.clip(rand_move, self.bounds[:, 0], self.bounds[:, 1])

    def swarm(self, i: int) -> Tuple[np.ndarray, bool]:
        """
        Swarming behavior: Fish moves toward local center of neighboring companions
        if food density is high and not overly crowded.
        """
        xi = self.population[i]
        fi = self.fitness[i]
        
        # Find neighbors within visual radius
        dists = np.linalg.norm(self.population - xi, axis=1)
        neighbors_idx = np.where((dists > 1e-8) & (dists <= self.visual))[0]
        nf = len(neighbors_idx)
        
        if nf == 0:
            return xi, False
            
        # Center of companions
        xc = np.mean(self.population[neighbors_idx], axis=0)
        fc = self._eval(xc)
        
        # Check if center is better and not overcrowded: fc / nf > delta * fi
        condition = (fc / nf) > (self.delta * fi) if self.maximize else (fc * nf) < (self.delta * fi)
        if condition and self._is_better(fc, fi):
            direction = (xc - xi) / (np.linalg.norm(xc - xi) + 1e-8)
            step_move = xi + direction * self.step_size * np.random.uniform(0.5, 1.0)
            return np.clip(step_move, self.bounds[:, 0], self.bounds[:, 1]), True
            
        return xi, False

    def chase(self, i: int) -> Tuple[np.ndarray, bool]:
        """
        Chasing behavior: Fish follows the neighbor with highest food density in visual field.
        """
        xi = self.population[i]
        fi = self.fitness[i]
        
        dists = np.linalg.norm(self.population - xi, axis=1)
        neighbors_idx = np.where((dists > 1e-8) & (dists <= self.visual))[0]
        nf = len(neighbors_idx)
        
        if nf == 0:
            return xi, False
            
        # Best companion
        neighbor_fitnesses = self.fitness[neighbors_idx]
        best_neighbor_local_idx = np.argmax(neighbor_fitnesses) if self.maximize else np.argmin(neighbor_fitnesses)
        best_neighbor_idx = neighbors_idx[best_neighbor_local_idx]
        
        xmax = self.population[best_neighbor_idx]
        fmax = self.fitness[best_neighbor_idx]
        
        condition = (fmax / nf) > (self.delta * fi) if self.maximize else (fmax * nf) < (self.delta * fi)
        if condition and self._is_better(fmax, fi):
            direction = (xmax - xi) / (np.linalg.norm(xmax - xi) + 1e-8)
            step_move = xi + direction * self.step_size * np.random.uniform(0.5, 1.0)
            return np.clip(step_move, self.bounds[:, 0], self.bounds[:, 1]), True
            
        return xi, False

    def step(self):
        """Execute one full generation of AFSA optimization"""
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            # Attempt Swarm and Chase behaviors
            x_swarm, swarm_success = self.swarm(i)
            x_chase, chase_success = self.chase(i)
            
            if swarm_success and chase_success:
                f_s = self._eval(x_swarm)
                f_c = self._eval(x_chase)
                new_population[i] = x_swarm if self._is_better(f_s, f_c) else x_chase
            elif swarm_success:
                new_population[i] = x_swarm
            elif chase_success:
                new_population[i] = x_chase
            else:
                # Default to Preying
                new_population[i] = self.prey(i)
                
        self.population = new_population
        for i in range(self.pop_size):
            self.fitness[i] = self._eval(self.population[i])
            if self._is_better(self.fitness[i], self.best_fitness):
                self.best_fitness = self.fitness[i]
                self.best_x = self.population[i].copy()
                
        self.history.append(self.best_fitness)

    def optimize(self, max_iter: int = 30) -> Tuple[np.ndarray, float, List[float]]:
        """Run AFSA optimization loop"""
        for _ in range(max_iter):
            self.step()
        return self.best_x, self.best_fitness, self.history
