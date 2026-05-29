from typing import Any, Dict, List, Optional, Tuple, Union, Set
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import haversine_distances


class GroundSet:
    """Dataset for submodular maximization with an integrated Knapsack Constraint.

    Attributes:
        costs (Dict[Any, float]): Normalized mapping of elements to their costs.
        capacity (float): The total available budget for the knapsack (normalized to 1.0).
        nb_elements (int): Total count of elements within the ground set.
        k (int): Maximum possible number of elements achievable within the budget.
        aux_data (Optional[Any]): Auxiliary data structures passed along to algorithms.
    """

    def __init__(
        self,
        costs: Dict[Any, float],
        capacity: float = 1.0,
        aux_data: Optional[Any] = None
    ):
        self.capacity = capacity
        # Initialize costs; default to 1.0 if not provided
        self.costs = self.normalize_costs(costs)
        self.elements = set(self.costs.keys())
        self.nb_elements = len(self.elements)
        self.k = self.get_maximal_elements()
        self.aux_data = aux_data

    def normalize_costs(self, costs: Dict[Any, float]) -> Dict[Any, float]:
        normalized = {
            e: c / self.capacity for e, c in costs.items() if c <= self.capacity
        }
        self.capacity = 1.0
        return normalized

    def get_maximal_elements(self) -> int:
        # Extract costs and sort them from smallest to largest
        costs = sorted(self.costs.values())
        total_cost = 0.0
        count = 0

        for cost in costs:
            if total_cost + cost <= self.capacity:
                total_cost += cost
                count += 1
            else:
                break

        return count

    def get_cost(self, e: Any) -> float:
        """Returns the cost of a specific element."""
        return self.costs.get(e, 1.0)

    def get_total_cost(self, S: Set[Any]) -> float:
        """Calculates combined structural cost of a given candidate subset."""
        return sum(self.get_cost(el) for el in S)

    def is_feasible(self, S: Set[Any]) -> bool:
        """Checks if adding element 'e' to set 'S' exceeds the knapsack capacity."""
        return self.get_total_cost(S) <= self.capacity

    def __repr__(self) -> str:
        return f"GroundSet(n={self.nb_elements}, capacity={self.capacity})"


# ==============================================================================
#  OBJECTIVES
# ==============================================================================

class UberMonotoneObjective:

    def __init__(
        self,
        passenger_coords: np.ndarray,
        grid_coords: np.ndarray,
        sensitivity: float,
        residual: Optional[Set[Any]] = None
    ):
        self.passengers = passenger_coords
        self.grid = grid_coords
        self.sensitivity = sensitivity
        self.num_queries = 0

        self.num_individuals = len(passenger_coords)

        self.residual = set(residual) if residual else set()

        self.residual_max_sims = np.zeros(self.num_individuals)
        for idx in self.residual:
            self.residual_max_sims = np.maximum(
                self.residual_max_sims, self._get_similarity_row(idx)
            )

        self.residual_value = np.sum(self.residual_max_sims) / self.num_individuals

    def _get_similarity_row(self, grid_idx: int) -> np.ndarray:
        """Calculates Manhattan convenience scores between passengers and a facility."""
        diffs = np.abs(self.passengers - self.grid[grid_idx])
        d_ab = np.sum(diffs, axis=1)
        convenience_scores = 2.0 - (2.0 / (1.0 + np.exp(-200.0 * d_ab)))
        return convenience_scores

    def evaluate(self, S: Set[Any], distort: bool = True) -> Tuple[float, np.ndarray]:
        """Performs full computational evaluation of subset S."""
        self.num_queries += 1

        # Start from the pre-computed residual state!
        max_sims = np.copy(self.residual_max_sims)

        new_items = set(S) - self.residual
        for idx in new_items:
            max_sims = np.maximum(max_sims, self._get_similarity_row(idx))

        coverage = np.sum(max_sims) / self.num_individuals

        S_val = coverage - self.residual_value
        return S_val, max_sims

    def marginal_gain(
        self,
        e: Any,
        auxiliary: np.ndarray,
        charge: bool = True
    ) -> Tuple[float, np.ndarray]:
        """Calculates utility step increase if element e is appended to S."""
        if charge:
            self.num_queries += 1

        max_sims = auxiliary

        sim_e = self._get_similarity_row(e)
        new_max_sims = np.maximum(max_sims, sim_e)

        coverage_gain = (np.sum(new_max_sims) - np.sum(max_sims)) / self.num_individuals

        return coverage_gain, new_max_sims

    def add_one_element(self, e: Any, S: Set[Any], auxiliary: np.ndarray) -> np.ndarray:
        """Updates and flushes auxiliary coverage tracking vector arrays."""
        max_sims = auxiliary
        sim_e = self._get_similarity_row(e)
        new_max_sims = np.maximum(max_sims, sim_e)
        return new_max_sims


class UberCutObjective:

    def __init__(
        self,
        passenger_coords: np.ndarray,
        grid_coords: Union[np.ndarray, Dict[int, Any]],
        sensitivity: float,
        scale: float,
        k: int,
        residual: Optional[Set[Any]] = None
    ):
        self.passengers = passenger_coords
        self.grid = grid_coords if isinstance(grid_coords, dict) else dict(enumerate(grid_coords))
        self.n = len(self.grid)
        self.sensitivity = sensitivity
        self.num_queries = 0
        self.num_individuals = len(passenger_coords)
        self.scale = scale
        self.k = k

        self.residual = set(residual) if residual else set()

        self.residual_max_sims = np.zeros(self.num_individuals)
        for idx in self.residual:
            self.residual_max_sims = np.maximum(
                self.residual_max_sims, self._get_similarity_row(idx)
            )

        self.total_sim_to_N = {}
        for idx in self.grid:
            # Sum of similarity from 'idx' to all 'k' in N
            self.total_sim_to_N[idx] = sum(self._get_facility_similarity(idx, k) for k in self.grid)

    def _get_similarity_row(self, grid_idx: int) -> np.ndarray:
        diffs = np.abs(self.passengers - self.grid[grid_idx])
        d_ab = np.sum(diffs, axis=1)
        return 2.0 - (2.0 / (1.0 + np.exp(-200.0 * d_ab)))

    def _get_facility_similarity(self, idx1: int, idx2: int) -> float:
        dist = np.sum(np.abs(self.grid[idx1] - self.grid[idx2]))
        return 2.0 - (2.0 / (1.0 + np.exp(-200.0 * dist)))

    def marginal_gain(
        self,
        e: Any,
        S: Set[Any],
        auxiliary: np.ndarray,
        charge: bool = True
    ) -> Tuple[float, np.ndarray]:
        if charge:
            self.num_queries += 1

        max_sims = auxiliary
        sim_e = self._get_similarity_row(e)
        new_max_sims = np.maximum(max_sims, sim_e)
        coverage_gain = (np.sum(new_max_sims) - np.sum(max_sims)) / self.num_individuals

        S_full = set(S).union(self.residual)

        sum_existing_sims = 0.0
        for u in S_full:
            sum_existing_sims += self._get_facility_similarity(u, e)

        self_sim = self._get_facility_similarity(e, e)
        total_sim_N = self.total_sim_to_N[e]

        cut_gain_raw = total_sim_N - self_sim - (2.0 * sum_existing_sims)

        diversity_gain = self.scale * (cut_gain_raw / (self.n * self.k))

        total_gain = coverage_gain + diversity_gain

        return total_gain, new_max_sims

    def add_one_element(self, e: Any, S: Set[Any], auxiliary: np.ndarray) -> np.ndarray:
        """Updates and merges lookahead execution tracking vectors."""
        max_sims = auxiliary
        sim_e = self._get_similarity_row(e)
        return np.maximum(max_sims, sim_e)

    def evaluate(self, S: Set[Any]) -> Tuple[float, np.ndarray]:

        self.num_queries += 1

        X = set(S).union(self.residual)
        if not X:
            return 0.0, np.zeros(self.num_individuals)

        # --- 1. Coverage Term ---
        max_sims = np.copy(self.residual_max_sims)
        new_in_S = set(S) - self.residual
        for idx in new_in_S:
            max_sims = np.maximum(max_sims, self._get_similarity_row(idx))

        coverage_val = np.sum(max_sims) / self.num_individuals

        leftovers = set(self.grid.keys()) - X

        diversity_sum = 0.0
        for u in X:
            for v in leftovers:
                diversity_sum += self._get_facility_similarity(u, v)

        diversity_val = diversity_sum / (self.n * self.k)

        return coverage_val + (self.scale * diversity_val), max_sims

class UberOptimizer:
    def __init__(self, points: Union[List[Tuple[float, float]], np.ndarray], n_data: int):
        """Initializes the optimizer with a Convex Hull boundary."""
        self.n_data = n_data

        # 1. Ensure points are a 2D numpy array (N, 2)
        self.points = np.asarray(points)
        if self.points.ndim == 1:
            # Safety for flat lists
            self.points = self.points.reshape(-1, 2)

        self.hull = ConvexHull(self.points)
        self.A = self.hull.equations[:, :2]  # Coefficients [a, b]
        self.b = self.hull.equations[:, 2]   # Constant [c]

    def is_inside(self, pts_array: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Vectorized check determining if point coordinates sit inside the hull boundary.
        """
        return np.all(self.A @ pts_array.T + self.b[:, None] <= tol, axis=0)

    def create_grid(self, n_locs, spurious):
        """
        Guarantees exactly n_locs.
        Adjusts grid spacing (delta) to ensure n_real points are evenly
        distributed specifically inside the polygon.
        """
        n_real = n_locs - spurious
        north_pole = self.points[np.argmax(self.points[:, 0])]

        if n_real <= 0:
            return np.tile(north_pole, (n_locs, 1))

        # 1. Get Bounding Box
        min_lat, min_lon = self.points.min(axis=0)
        max_lat, max_lon = self.points.max(axis=0)

        # 2. Estimate initial spacing (Delta)
        # Area of bounding box / target points gives an approximate cell size
        area_approx = (max_lat - min_lat) * (max_lon - min_lon)
        # We use a slightly smaller delta because the polygon is smaller than the box
        delta = np.sqrt(area_approx / (n_real * 2))

        real_pts = []
        # We iterate a few times to fine-tune the density if the polygon is weird
        for _ in range(3):
            lats = np.arange(min_lat, max_lat, delta)
            lons = np.arange(min_lon, max_lon, delta)
            lat_grid, lon_grid = np.meshgrid(lats, lons)
            candidates = np.c_[lat_grid.ravel(), lon_grid.ravel()]

            # Filter by polygon
            mask = self.is_inside(candidates)
            valid = candidates[mask]

            if len(valid) >= n_real:
                # We found enough! Keep these and break
                real_pts = valid
                break
            else:
                # Not enough points landed in the polygon, make the grid denser
                delta *= 0.8

         # 3. Evenly Downsample the valid set to exactly n_real
        # This maintains the "grid" alignment while hitting the exact count
        if len(real_pts) > n_real:
            indices = np.linspace(0, len(real_pts) - 1, n_real, dtype=int)
            real_pts = real_pts[indices]
        elif len(real_pts) < n_real:
            # Emergency fallback
            print('Not enough')
            padding = n_real - len(real_pts)
            real_pts = np.vstack([real_pts, np.tile(north_pole, (padding, 1))])

        # 4. Add Spurious
        spurious_pts = np.tile(north_pole, (spurious, 1))

        grid = np.vstack([real_pts, spurious_pts])
        return grid

    def process_raw_data(self, input_csv: str, output_csv: str) -> np.ndarray:
        """Filters Uber data entries against geometric arrays to gather exact data targets."""
        all_valid_points = []

        # 1. Collect ALL points that fall inside the polygon
        print("Filtering points inside hull...")
        for chunk in pd.read_csv(input_csv, chunksize=50000):
            # Taking Lat/Lon columns
            chunk_pts = chunk.iloc[:, [1, 2]].values

            # Filter check
            mask = self.is_inside(chunk_pts)
            valid_chunk = chunk_pts[mask]

            if len(valid_chunk) > 0:
                all_valid_points.append(valid_chunk)

        # Flatten the list of arrays into one large matrix
        if not all_valid_points:
            raise ValueError("No points found inside the provided hull.")

        all_valid_points = np.vstack(all_valid_points)
        total_found = len(all_valid_points)
        print(f"Found {total_found} valid points in total.")

        # 2. Select EXACTLY self.n_data points
        if total_found >= self.n_data:
            # Use random indices to get exactly the count requested
            indices = np.random.choice(total_found, int(self.n_data), replace=False)
            processed_data = all_valid_points[indices]
        else:
            # If we have fewer than n_data, we take everything
            print(f"Warning: Only {total_found} points found. Returning all.")
            processed_data = all_valid_points

        # 3. Save and return
        df_out = pd.DataFrame(processed_data, columns=['lat', 'lon'])
        df_out.to_csv(output_csv, index=False)

        return df_out.values

    def read_from_file(self, output_csv: str) -> np.ndarray:
        df = pd.read_csv(output_csv)
        return df.values

    def assign_costs_km(
        self,
        grid_coords: np.ndarray,
        poi_coords: Tuple[float, float] = (40.7505, -73.9934)
    ) -> Dict[int, float]:
        """Calculates the Haversine distance in KM to map real-estate inverse costs."""
        grid_radians = np.radians(grid_coords)
        poi_radians = np.radians(np.array([poi_coords]))

        # 2. Calculate the Haversine distance
        distances_radians = haversine_distances(grid_radians, poi_radians).flatten()

        distances_km = distances_radians * 6371.0

        base_cost = 1.0
        premium_cost = 19.0
        real_estate_costs = base_cost + (premium_cost / np.maximum(distances_km, 1.0))

        return dict(enumerate(real_estate_costs))


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":

    # Manhattan conex hull
    full_island_hull = [
        (40.7005038, -74.0144209), (40.7112088, -73.9776851),
        (40.7282434, -73.9720702), (40.7418214, -73.9733576),
        (40.7754746, -73.9430232), (40.7974885, -73.9296695),
        (40.8350989, -73.9354202), (40.8713327, -73.9109482),
        (40.8769142, -73.9269985), (40.8512745, -73.9448513),
        (40.7607748, -74.0040745), (40.7474382, -74.0115323),
        (40.7125758, -74.0182271)
    ]

    opt = UberOptimizer(full_island_hull, n_data=20000)
