from optimal_point_gpu import OptimalPointGPU
from euclidean_distance import Distance 

class OptimalPointDiCEGPU(): 

    def __init__(self, dataset, model): 
        self.optimal_point = OptimalPointGPU(
            dataset=dataset, 
            model=model
        )
        self.distance_tool = Distance()

    def run(self, desired_class, original_class, chosen_row=-1, threshold=10000, point_epsilon=1e-3, epsilon=0.01, constraints=[], deltas=[], plot=False): 

        final_optimal_pt = self.optimal_point.run(desired_class, original_class, 
                                chosen_row, threshold,
                                point_epsilon, epsilon,
                                constraints, deltas, plot)
        query_instance = self.optimal_point.get_undesired_datapt()
        optimal_datapt = self.optimal_point.get_optimal_datapt()
        boundary_points = self.optimal_point.get_boundary_points()

        return self.dataset, self.model, query_instance, final_optimal_pt, self.distance_tool.euclidean_distance(query_instance, optimal_datapt), deltas, boundary_points