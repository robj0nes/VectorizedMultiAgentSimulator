import torch
from torch_bp.bp import LoopyLinearGaussianBP
from torch_bp.graph import FactorGraph
from torch_bp.graph.factors.linear_gaussian_factors import UnaryGaussianLinearFactor, PairwiseGaussianLinearFactor


# TODO: Refactor so that the agent anchor factors correspond to the name index.

class GaussianBeliefPropogation():
    '''
    Takes a `graph_dict` which defines the nodes groups and relationships within the graph.
    Example where we assume node[0] is the robot hosting the graph:
        graph_dict = {
                    'robots': {
                        'nodes': [0, 1, 2, 3],
                        'edges': [[0, 1], [0, 2], [0, 3]]
                    },
                    'goals': {
                        'nodes': [4, 5, 6, 7],
                        'edges': [[0, 4], [0, 5], [0, 6], [0, 7]]
                    }
                }
    '''

    def __init__(self,
                 graph_dict: dict,
                 batch_dim: int,
                 device: torch.device,
                 msg_passing_iters: int = 1,
                 msgs_per_iter: int = 1,
                 dtype: torch.dtype = torch.float64):
        self.batch_dim = batch_dim
        self.device = device
        self.dtype = dtype
        self.graph_dict = graph_dict
        self.msg_passing_iters = msg_passing_iters
        self.msgs_per_iter = msgs_per_iter
        # TODO: Store lists of node indices for agents, goals and otherwise.

        self.total_nodes = 0
        for k in self.graph_dict.keys():
            self.total_nodes += len(graph_dict[k]['nodes'])

        self.sigma = 0.05

        # Define initial mu and covariance for nodes.
        self.init_node_mu = ((torch.randn(self.batch_dim, self.total_nodes, 2) * 2 + 2)
                             .to(device=self.device, dtype=self.dtype))
        self.init_node_covar = (2 * torch.eye(2, device=self.device, dtype=self.dtype)
                                .repeat(self.batch_dim, self.total_nodes, 1, 1))

        # Define z_bias and covars for pairwise distance factors.
        self.init_dist_z_bias = (1.0 * torch.ones(1, device=self.device, dtype=self.dtype)
                                 .unsqueeze(0).repeat(self.batch_dim, 1))
        self.init_dist_covar = (2 * self.sigma * torch.eye(1, device=self.device, dtype=self.dtype)
                                .unsqueeze(0).repeat(self.batch_dim, 1, 1))

        self.initialise_factor_graph()
        self.gbp = self.initialise_gbp()

        self.current_means = None
        self.current_covars = None

    def unary_anchor_fn(self, x: torch.Tensor):
        '''Default unary anchor function.'''
        batch_shape = x.shape[:1]
        grad = torch.eye(x.shape[-1], dtype=x.dtype).repeat(batch_shape + (1, 1))
        return grad, x

    def pairwise_dist_fn(self, x: torch.Tensor):
        '''Default pairwise distance function.'''
        h_fn = lambda x: torch.linalg.norm(x[:, 2:] - x[:, :2], dim=-1)
        grad_fn = torch.func.jacrev(h_fn)
        grad_x = grad_fn(x).diagonal(dim1=0, dim2=1).transpose(1, 0)
        return grad_x[:, None, :], h_fn(x).unsqueeze(-1)

    def initialise_gbp(self) -> LoopyLinearGaussianBP:
        fac_grap = FactorGraph(num_nodes=self.total_nodes,
                               factors=self.factors,
                               factor_neighbours=self.factor_neighbours)

        return LoopyLinearGaussianBP(node_means=self.init_node_mu, node_covars=self.init_node_covar,
                                     factor_graph=fac_grap,
                                     tensor_kwargs={'device': self.device,
                                                    'dtype': torch.float64},
                                     batch_dim=self.batch_dim)

    def initialise_factor_graph(self):
        """Initialise a base FG to assign to all robots. This implementation defines factors between
        robots own positions, other robot positions, and goal positions."""
        # Note:
        #   - We fix the total variables (nodes) for the factor graph according to the task needs
        #   - Initialise the graph with 'zero' factors between known relationships
        #   - As the task evolves we introduce factors where necessary,
        #       but must be introduced across all batched envs (batch_dim)
        self.factors = []
        self.factor_neighbours = []
        # First we create anchor factors for all variables. Everything is initialised with zero means.
        anchor_factors = [UnaryGaussianLinearFactor(self.unary_anchor_fn,
                                                    torch.zeros(self.batch_dim, 2, device=self.device,
                                                                dtype=self.dtype),
                                                    self.sigma * torch.eye(2, device=self.device, dtype=torch.float64)
                                                    .unsqueeze(0).repeat(self.batch_dim, 1, 1),
                                                    self.init_node_mu[:, i],
                                                    True)
                          for i in range(self.total_nodes)]
        self.factor_neighbours.extend([(i,) for i in range(self.total_nodes)])
        self.factors.extend(anchor_factors)

        for key in self.graph_dict.keys():
            dist_factors = [
                PairwiseGaussianLinearFactor(
                    self.pairwise_dist_fn,
                    self.init_dist_z_bias,
                    self.init_dist_covar,
                    torch.concat(
                        (self.init_node_mu[:, i],
                         self.init_node_mu[:, j]),
                        dim=-1)
                    .to(self.device),
                    False
                )
                for edge in self.graph_dict[key]['edges'] for i, j in [edge]
            ]
            self.factor_neighbours.extend([(i, j) for edge in self.graph_dict[key]['edges'] for i, j in [edge]])
            self.factors.extend(dist_factors)

    def update_anchor(self, x: torch.Tensor, anchor_index: int, env_index=None):
        self.gbp.factor_graph.factor_clusters[anchor_index].factor.update_bias(x, batch_dim=env_index)

    def iterate_gbp(self):
        self.current_means, self.current_covars = self.gbp.solve(num_iters=self.msg_passing_iters,
                                                                 msg_pass_per_iter=self.msgs_per_iter)
        # Note: probably not required..
        self.vars = torch.diagonal(self.current_covars, dim1=-2, dim2=-1)
        self.stds = torch.sqrt(self.vars)
