

class Dataset:
    def __init__(self, n_problem_inputs: int, n_problem_outputs: int, n_patterns: int):
        self.k = n_problem_inputs
        self.J = n_problem_outputs
        self.N = n_patterns