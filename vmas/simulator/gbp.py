import numpy as np
import torch
import torch_bp.distributions as dist
from torch_bp.bp import LoopyLinearGaussianBP
from torch_bp.graph import FactorGraph
from torch_bp.graph.factors.linear_gaussian_factors import UnaryGaussianLinearFactor, PairwiseGaussianLinearFactor


class GaussianBeliefPropagation:
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

        self.total_nodes = 0
        self.node_tags = []
        for k in self.graph_dict.keys():
            self.node_tags.extend([k] * len(graph_dict[k]['nodes']))
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

    def initialise_gbp(self, node_means=None, node_covars=None) -> LoopyLinearGaussianBP:
        fac_grap = FactorGraph(num_nodes=self.total_nodes,
                               factors=self.factors,
                               factor_neighbours=self.factor_neighbours,
                               node_tags=self.node_tags)

        return LoopyLinearGaussianBP(node_means=self.init_node_mu if node_means is None else node_means,
                                     node_covars=self.init_node_covar if node_covars is None else node_covars,
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
        # First we create anchor factors for all variables. Everything is initialised with zero bias. x_0 is random.
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
            if 'edges' in self.graph_dict[key].keys():
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

    def add_node(self, factors, factor_neighbours, node_tag, node_mean):
        self.gbp.node_means = torch.cat((self.gbp.node_means, node_mean.unsqueeze(0)), dim=1).to(
            self.device)
        self.gbp.node_covars = torch.cat((self.gbp.node_covars, self.init_node_covar), dim=1).to(
            self.device)
        node_id, factor_id = self.gbp.factor_graph.add_node(factors, factor_neighbours, node_tag)

        # TODO: Need to figure out how to correctly instantiate a new node and factor in the msg DB.,
        self.gbp._precompute_factors()
        for n_id in factor_neighbours:
            self.gbp.msg_factor_to_node_db[(factor_id, n_id)] = (
                self.gbp._compute_msg_from_factor(self.gbp.factor_graph.factor_clusters[factor_id], n_id))
            self.gbp.msg_node_to_factor_db[(n_id, factor_id)] = (
                self.gbp._compute_msg_from_node(n_id, factor_id))

    def update_anchor(self, x: torch.Tensor, anchor_index: int, env_index=None):
        self.gbp.factor_graph.factor_clusters[anchor_index].factor.update_bias(new_z=x, batch_dim=env_index)

    def update_node(self, new_mu: torch.Tensor, new_covar: torch.Tensor, node_index: int, env_index=None):
        if env_index is None:
            self.gbp.node_means[:, node_index] = new_mu
            self.gbp.node_covars[:, node_index] = new_covar
            self.update_anchor(new_mu, node_index, env_index)
        else:
            self.gbp.node_means[env_index, node_index] = new_mu
            self.gbp.node_covars[env_index, node_index] = new_covar
            self.update_anchor(new_mu, node_index, env_index)

    def remake_graph(self):
        # TODO: This is overwriting the new node_means and covars..
        self.gbp = self.initialise_gbp(node_means=self.current_means, node_covars=self.current_covars)

    def replace_factor_at(self, factor, index_in, index_out):
        self.factors.insert(index_in, factor)
        self.factors.pop(index_out)

    def iterate_gbp(self):
        self.current_means, self.current_covars = self.gbp.solve(num_iters=self.msg_passing_iters,
                                                                 msg_pass_per_iter=self.msgs_per_iter)
        # Note: probably not required..
        self.vars = torch.diagonal(self.current_covars, dim1=-2, dim2=-1)
        self.stds = torch.sqrt(self.vars)

    def get_gaussian_ellipses(self, env_index: int):
        gaussians = [(mu, sigma) for mu, sigma in zip(self.gbp.node_means[env_index],
                                                      self.gbp.node_covars[env_index])]
        edge_pairs = [m for m in self.factor_neighbours if len(m) > 1]
        edge_coords = [(self.gbp.node_means[env_index][m1], self.gbp.node_means[env_index][m2]) for (m1, m2) in
                       edge_pairs]
        std_devs = [x * 0.2 for x in range(0, 10)]
        all_ellipses = []
        for i, (mu, sigma) in enumerate(gaussians):
            ellipses = []
            for j in std_devs:
                # Eigenvalues and eigenvectors of the covariance matrix
                eigenvalues, eigenvectors = torch.linalg.eigh(sigma)

                # Radii are proportional to the square root of the eigenvalues (scaled by std_devs)
                radii_x, radii_y = j * torch.sqrt(eigenvalues)
                # semi_minor, semi_major = j * torch.sqrt(eigenvalues)

                # Note: The last eigenvalue/eigenvector are the largest, get angle wrt. x-axis.
                # minor_rot = torch.atan2(eigenvectors[0][1], eigenvectors[0][0])
                # major_rot = torch.atan2(eigenvectors[1][1], eigenvectors[1][0])
                rot_angle = torch.atan2(eigenvectors[-1][1], eigenvectors[-1][0])
                ellipses.append({
                    'mean': mu,
                    'radius': (radii_x, radii_y),
                    'rot_angle': rot_angle
                })
            all_ellipses.append(ellipses)
        return all_ellipses, edge_coords

    def get_gaussian_grid_sample(self, env_index: int, sample_size: int = 2, n_samples: int = 200):
        gaussians = [dist.Gaussian(mu, sigma, device=self.device)
                     for mu, sigma in zip(self.current_means[env_index],
                                          self.current_covars[env_index])]

        all_gaussians = []
        for g in gaussians:
            # Eval gaussian grid-wise in worldspace and collect any pdf(x) > 2
            np.set_printoptions(legacy='1.25')
            X, Y, Z = g.eval_grid([-sample_size, sample_size, -sample_size, sample_size],
                                  n_samples=n_samples)
            ys, xs = np.where(Z > 0.5)
            # Extract the world coordinates at each evaulation point. TODO: More thorough testing that this is correct.
            marker_pos = [(X[0][xs[i]], Y[ys[i]][0], Z[ys[i]][xs[i]]) for i in range(len(xs))]

            sample = []
            if len(marker_pos) > 0:
                # Normalise the z values for better rendering.
                zs = [m[2] for m in marker_pos]
                min_zs = np.min(zs)
                max_zs = np.max(zs)
                norm_markers = [(mp[0], mp[1], (mp[2] - min_zs) / (max_zs - min_zs)) for mp in marker_pos]
                sample.extend(norm_markers)
            all_gaussians.append(sample)
        return all_gaussians
