#' AGNNConv
#' 
#' @description \loadmathjax 
#' An Attention-based Graph Neural Network (AGNN) as presented by
#' [Thekumparampil et al. (2018)](https://arxiv.org/abs/1803.03735).
#' 
#' **Mode**: single, disjoint.
#' 
#' **This layer expects a sparse adjacency matrix.**
#' 
#' This layer computes:
#' \mjdeqn{ Z = PX }{}
#' where
#' \mjdeqn{ P_{ij} = \frac{ \exp \left( \beta \cos \left( X_i, X_j \right) \right) }{ \sum\limits_{k \in \mathcal{N}(i) \cup \{ i \}} \exp \left( \beta \cos \left( X_i, X_k \right) \right) } }{}
#' and \mjeqn{\beta}{} is a trainable parameter.
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`;
#' - Binary adjacency matrix of shape `(N, N)`.
#' 
#' **Output**
#' 
#' - Node features with the same shape of the input.
#' 
#' 
#' @param trainable boolean, if True, then beta is a trainable parameter.
#' Otherwise, beta is fixed to 1
#' @param activation activation function to use
#' @export
layer_agnn_conv <- function(object,
	trainable = TRUE,
	activation = NULL,
	...)
{
	args <- list(trainable = trainable,
		activation = activation
		)
	keras::create_layer(spk$layers$AGNNConv, object, args)
}

#' APPNP
#' 
#' @description \loadmathjax 
#' A graph convolutional layer implementing the APPNP operator, as presented by
#' [Klicpera et al. (2019)](https://arxiv.org/abs/1810.05997).
#' 
#' This layer computes:
#' \mjdeqn{ Z^{(0)} = \textrm{MLP}(X); \\ Z^{(K)} = (1 - \alpha) \hat D^{-1/2} \hat A \hat D^{-1/2} Z^{(K - 1)} + \alpha Z^{(0)}, }{}
#' where \mjeqn{\alpha}{} is the _teleport_ probability and \mjeqn{\textrm{MLP}}{} is a
#' multi-layer perceptron.
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Modified Laplacian of shape `([batch], N, N)`; can be computed with
#' `spektral.utils.convolution.localpooling_filter`.
#' 
#' **Output**
#' 
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#' 
#' 
#' @param channels number of output channels
#' @param alpha teleport probability during propagation
#' @param propagations number of propagation steps
#' @param mlp_hidden list of integers, number of hidden units for each hidden
#' layer in the MLP (if None, the MLP has only the output layer)
#' @param mlp_activation activation for the MLP layers
#' @param dropout_rate dropout rate for Laplacian and MLP layers
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_appnp <- function(object,
	channels,
	alpha = 0.2,
	propagations = 1,
	mlp_hidden = NULL,
	mlp_activation = 'relu',
	dropout_rate = 0.0,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		alpha = alpha,
		propagations = as.integer(propagations),
		mlp_hidden = as.integer(mlp_hidden),
		mlp_activation = mlp_activation,
		dropout_rate = dropout_rate,
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$APPNP, object, args)
}

#' ARMAConv
#' 
#' @description \loadmathjax 
#' A graph convolutional layer with \mjeqn{\mathrm{ARMA} _ K}{} filters, as presented by
#' [Bianchi et al. (2019)](https://arxiv.org/abs/1901.01343).
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' This layer computes:
#' \mjdeqn{ Z = \frac{1}{K} \sum\limits_{k=1}^K \bar X_k^{(T)}, }{}
#' where \mjeqn{K}{} is the order of the \mjeqn{\mathrm{ARMA} _ K}{} filter, and where:
#' \mjdeqn{ \bar X_k^{(t + 1)} = \sigma \left(\tilde L \bar X^{(t)} W^{(t)} + X V^{(t)} \right) }{}
#' is a recursive approximation of an \mjeqn{\mathrm{ARMA} _ 1}{} filter, where
#' \mjeqn{ \bar X^{(0)} = X }{}
#' and
#' \mjdeqn{ \tilde L =  \frac{2}{\lambda_{max}} \cdot (I - D^{-1/2} A D^{-1/2}) - I }{}
#' is the normalized Laplacian with a rescaled spectrum.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Normalized and rescaled Laplacian of shape `([batch], N, N)`; can be
#' computed with `spektral.utils.convolution.normalized_laplacian` and
#' `spektral.utils.convolution.rescale_laplacian`.
#' 
#' **Output**
#' 
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#' 
#' 
#' @param channels number of output channels
#' @param order order of the full ARMA\(_K\) filter, i.e., the number of parallel
#' stacks in the layer
#' @param iterations number of iterations to compute each ARMA\(_1\) approximation
#' @param share_weights share the weights in each ARMA\(_1\) stack.
#' @param gcn_activation activation function to use to compute each ARMA\(_1\)
#' stack
#' @param dropout_rate dropout rate for skip connection
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_arma_conv <- function(object,
	channels,
	order = 1,
	iterations = 1,
	share_weights = FALSE,
	gcn_activation = 'relu',
	dropout_rate = 0.0,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		order = as.integer(order),
		iterations = as.integer(iterations),
		share_weights = share_weights,
		gcn_activation = gcn_activation,
		dropout_rate = dropout_rate,
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$ARMAConv, object, args)
}

#' ChebConv
#' 
#' @description \loadmathjax 
#' A Chebyshev convolutional layer as presented by
#' [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' This layer computes:
#' \mjdeqn{ Z = \sum \limits_{k=0}^{K - 1} T^{(k)} W^{(k)}  + b^{(k)}, }{}
#' where \mjeqn{ T^{(0)}, ..., T^{(K - 1)} }{} are Chebyshev polynomials of \mjeqn{\tilde L}{}
#' defined as
#' \mjdeqn{ T^{(0)} = X \\ T^{(1)} = \tilde L X \\ T^{(k \ge 2)} = 2 \cdot \tilde L T^{(k - 1)} - T^{(k - 2)}, }{}
#' where
#' \mjdeqn{ \tilde L =  \frac{2}{\lambda_{max}} \cdot (I - D^{-1/2} A D^{-1/2}) - I }{}
#' is the normalized Laplacian with a rescaled spectrum.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - A list of K Chebyshev polynomials of shape
#' `[([batch], N, N), ..., ([batch], N, N)]`; can be computed with
#' `spektral.utils.convolution.chebyshev_filter`.
#' 
#' **Output**
#' 
#' - Node features with the same shape of the input, but with the last
#' dimension changed to `channels`.
#' 
#' 
#' @param channels number of output channels
#' @param K order of the Chebyshev polynomials
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_cheb_conv <- function(object,
	channels,
	K = 1,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		K = as.integer(K),
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$ChebConv, object, args)
}

#' CrystalConv
#' 
#' @description \loadmathjax 
#' A Crystal Graph Convolutional layer as presented by
#' [Xie & Grossman (2018)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301).
#' 
#' **Mode**: single, disjoint.
#' 
#' **This layer expects a sparse adjacency matrix.**
#' 
#' This layer computes for each node \mjeqn{i}{}:
#' \mjdeqn{ H_i = X_i + \sum\limits_{j \in \mathcal{N}(i)} \sigma \left( z_{ij} W^{(f)} + b^{(f)} \right) \odot g \left( z_{ij} W^{(s)} + b^{(s)} \right) }{}
#' where \mjeqn{z_{ij} = X_i \| X_j \| E_{ij} }{}, \mjeqn{\sigma}{} is a sigmoid
#' activation, and \mjeqn{g}{} is the activation function (defined by the `activation`
#' argument).
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`;
#' - Binary adjacency matrix of shape `(N, N)`.
#' - Edge features of shape `(num_edges, S)`.
#' 
#' **Output**
#' 
#' - Node features with the same shape of the input, but the last dimension
#' changed to `channels`.
#' 
#' 
#' @param channels integer, number of output channels
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_crystal_conv <- function(object,
	channels,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$CrystalConv, object, args)
}

#' DiffuseFeatures
#' 
#' @description \loadmathjax Utility layer calculating a single channel of the
#' diffusional convolution.
#' 
#' Procedure is based on https://arxiv.org/abs/1707.01926
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Normalized adjacency or attention coef. matrix \mjeqn{\hat A }{} of shape
#' `([batch], N, N)`; Use DiffusionConvolution.preprocess to normalize.
#' 
#' **Output**
#' 
#' - Node features with the same shape as the input, but with the last
#' dimension changed to \mjeqn{1}{}.
#' 
#' 
#' @param num_diffusion_steps How many diffusion steps to consider. \(K\) in paper.
#' @param kernel_initializer initializer for the weights
#' @param kernel_regularizer regularization applied to the kernel vectors
#' @param kernel_constraint constraint applied to the kernel vectors
#' @export
layer_diffuse_features <- function(object,
	num_diffusion_steps,
	kernel_initializer,
	kernel_regularizer,
	kernel_constraint,
	...)
{
	args <- list(num_diffusion_steps = num_diffusion_steps,
		kernel_initializer = kernel_initializer,
		kernel_regularizer = kernel_regularizer,
		kernel_constraint = kernel_constraint
		)
	keras::create_layer(spk$layers$DiffuseFeatures, object, args)
}

#' DiffusionConv
#' 
#' @description \loadmathjax Applies Graph Diffusion Convolution as descibed by
#' [Li et al. (2016)](https://arxiv.org/pdf/1707.01926.pdf)
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' **This layer expects a dense adjacency matrix.**
#' 
#' Given a number of diffusion steps \mjeqn{K}{} and a row normalized adjacency matrix \mjeqn{\hat A }{},
#' this layer calculates the q'th channel as:
#' \mjdeqn{ \mathbf{H} _ {~:,~q} = \sigma\left( \sum_{f=1}^{F} \left( \sum_{k=0}^{K-1}\theta_k {\hat A}^k \right) X_{~:,~f} \right) }{}
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Normalized adjacency or attention coef. matrix \mjeqn{\hat A }{} of shape
#' `([batch], N, N)`; Use `DiffusionConvolution.preprocess` to normalize.
#' 
#' **Output**
#' 
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#' 
#' 
#' @param channels number of output channels
#' @param num_diffusion_steps How many diffusion steps to consider. \(K\) in paper.
#' @param kernel_initializer initializer for the weights
#' @param kernel_regularizer regularization applied to the weights
#' @param kernel_constraint constraint applied to the weights
#' @param activation activation function \(\sigma\) (\(\tanh\) by default)
#' @export
layer_diffusion_conv <- function(object,
	channels,
	num_diffusion_steps =  6,
	kernel_initializer = 'glorot_uniform',
	kernel_regularizer = NULL,
	kernel_constraint = NULL,
	activation = 'tanh',
	...)
{
	args <- list(channels = as.integer(channels),
		num_diffusion_steps = num_diffusion_steps,
		kernel_initializer = kernel_initializer,
		kernel_regularizer = kernel_regularizer,
		kernel_constraint = kernel_constraint,
		activation = activation
		)
	keras::create_layer(spk$layers$DiffusionConv, object, args)
}

#' EdgeConditionedConv
#' 
#' @description \loadmathjax 
#' An edge-conditioned convolutional layer (ECC) as presented by
#' [Simonovsky & Komodakis (2017)](https://arxiv.org/abs/1704.02901).
#' 
#' **Mode**: single, disjoint, batch.
#' 
#' **Notes**:
#' - This layer expects dense inputs and self-loops when working in batch mode.
#' - In single mode, if the adjacency matrix is dense it will be converted
#' to a SparseTensor automatically (which is an expensive operation).
#' 
#' For each node \mjeqn{ i }{}, this layer computes:
#' \mjdeqn{ Z_i = X_{i} W_{\textrm{root}} + \sum\limits_{j \in \mathcal{N}(i)} X_{j} \textrm{MLP}(E_{ji}) + b }{}
#' where \mjeqn{\textrm{MLP}}{} is a multi-layer perceptron that outputs an
#' edge-specific weight as a function of edge attributes.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Binary adjacency matrices of shape `([batch], N, N)`;
#' - Edge features. In single mode, shape `(num_edges, S)`; in batch mode, shape
#' `(batch, N, N, S)`.
#' 
#' **Output**
#' 
#' - node features with the same shape of the input, but the last dimension
#' changed to `channels`.
#' 
#' 
#' @param channels integer, number of output channels
#' @param kernel_network a list of integers representing the hidden neurons of
#' the kernel-generating network
#' @param root NA
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_edge_conditioned_conv <- function(object,
	channels,
	kernel_network = NULL,
	root = TRUE,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		kernel_network = kernel_network,
		root = root,
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$EdgeConditionedConv, object, args)
}

#' EdgeConv
#' 
#' @description \loadmathjax 
#' An Edge Convolutional layer as presented by
#' [Wang et al. (2018)](https://arxiv.org/abs/1801.07829).
#' 
#' **Mode**: single, disjoint.
#' 
#' **This layer expects a sparse adjacency matrix.**
#' 
#' This layer computes for each node \mjeqn{i}{}:
#' \mjdeqn{ Z_i = \sum\limits_{j \in \mathcal{N}(i)} \textrm{MLP}\big( X_i \| X_j - X_i \big) }{}
#' where \mjeqn{\textrm{MLP}}{} is a multi-layer perceptron.
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`;
#' - Binary adjacency matrix of shape `(N, N)`.
#' 
#' **Output**
#' 
#' - Node features with the same shape of the input, but the last dimension
#' changed to `channels`.
#' 
#' 
#' @param channels integer, number of output channels
#' @param mlp_hidden list of integers, number of hidden units for each hidden
#' layer in the MLP (if None, the MLP has only the output layer)
#' @param mlp_activation activation for the MLP layers
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_edge_conv <- function(object,
	channels,
	mlp_hidden = NULL,
	mlp_activation = 'relu',
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		mlp_hidden = as.integer(mlp_hidden),
		mlp_activation = mlp_activation,
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$EdgeConv, object, args)
}

#' GatedGraphConv
#' 
#' @description \loadmathjax 
#' A gated graph convolutional layer as presented by
#' [Li et al. (2018)](https://arxiv.org/abs/1511.05493).
#' 
#' **Mode**: single, disjoint.
#' 
#' **This layer expects a sparse adjacency matrix.**
#' 
#' This layer repeatedly applies a GRU cell \mjeqn{L}{} times to the node attributes
#' \mjdeqn{ \begin{align} & h^{(0)} _ i = X_i \| \mathbf{0} \\ & m^{(l)} _ i = \sum\limits_{j \in \mathcal{N}(i)} h^{(l - 1)} _ j W \\ & h^{(l)} _ i = \textrm{GRU} \left(m^{(l)} _ i, h^{(l - 1)} _ i \right) \\ & Z_i = h^{(L)} _ i \end{align} }{}
#' where \mjeqn{\textrm{GRU}}{} is the GRU cell.
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`; note that `F` must be smaller or equal
#' than `channels`.
#' - Binary adjacency matrix of shape `(N, N)`.
#' 
#' **Output**
#' 
#' - Node features with the same shape of the input, but the last dimension
#' changed to `channels`.
#' 
#' 
#' @param channels integer, number of output channels
#' @param n_layers integer, number of iterations with the GRU cell
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_gated_graph_conv <- function(object,
	channels,
	n_layers,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		n_layers = as.integer(n_layers),
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$GatedGraphConv, object, args)
}

#' GINConv
#' 
#' @description \loadmathjax 
#' A Graph Isomorphism Network (GIN) as presented by
#' [Xu et al. (2018)](https://arxiv.org/abs/1810.00826).
#' 
#' **Mode**: single, disjoint.
#' 
#' **This layer expects a sparse adjacency matrix.**
#' 
#' This layer computes for each node \mjeqn{i}{}:
#' \mjdeqn{ Z_i = \textrm{MLP}\big( (1 + \epsilon) \cdot X_i + \sum\limits_{j \in \mathcal{N}(i)} X_j \big) }{}
#' where \mjeqn{\textrm{MLP}}{} is a multi-layer perceptron.
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`;
#' - Binary adjacency matrix of shape `(N, N)`.
#' 
#' **Output**
#' 
#' - Node features with the same shape of the input, but the last dimension
#' changed to `channels`.
#' 
#' 
#' @param channels integer, number of output channels
#' @param epsilon unnamed parameter, see
#' [Xu et al. (2018)](https://arxiv.org/abs/1810.00826), and the equation above.
#' By setting `epsilon=None`, the parameter will be learned (default behaviour).
#' If given as a value, the parameter will stay fixed.
#' @param mlp_hidden list of integers, number of hidden units for each hidden
#' layer in the MLP (if None, the MLP has only the output layer)
#' @param mlp_activation activation for the MLP layers
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_gin_conv <- function(object,
	channels,
	epsilon = NULL,
	mlp_hidden = NULL,
	mlp_activation = 'relu',
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		epsilon = epsilon,
		mlp_hidden = as.integer(mlp_hidden),
		mlp_activation = mlp_activation,
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$GINConv, object, args)
}

#' GraphAttention
#' 
#' @description \loadmathjax 
#' A graph attention layer (GAT) as presented by
#' [Velickovic et al. (2017)](https://arxiv.org/abs/1710.10903).
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' **This layer expects dense inputs when working in batch mode.**
#' 
#' This layer computes a convolution similar to `layers.GraphConv`, but
#' uses the attention mechanism to weight the adjacency matrix instead of
#' using the normalized Laplacian:
#' \mjdeqn{ Z = \mathbf{\alpha}XW + b }{}
#' where
#' \mjdeqn{ \mathbf{\alpha} _ {ij} = \frac{ \exp\left( \mathrm{LeakyReLU}\left( a^{\top} [(XW)_i \, \| \, (XW)_j] \right) \right) } {\sum\limits_{k \in \mathcal{N}(i) \cup \{ i \}} \exp\left( \mathrm{LeakyReLU}\left( a^{\top} [(XW)_i \, \| \, (XW)_k] \right) \right) } }{}
#' where \mjeqn{a \in \mathbb{R}^{2F'}}{} is a trainable attention kernel.
#' Dropout is also applied to \mjeqn{\alpha}{} before computing \mjeqn{Z}{}.
#' Parallel attention heads are computed in parallel and their results are
#' aggregated by concatenation or average.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Binary adjacency matrix of shape `([batch], N, N)`;
#' 
#' **Output**
#' 
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`;
#' - if `return_attn_coef=True`, a list with the attention coefficients for
#' each attention head. Each attention coefficient matrix has shape
#' `([batch], N, N)`.
#' 
#' 
#' @param channels number of output channels
#' @param attn_heads number of attention heads to use
#' @param concat_heads bool, whether to concatenate the output of the attention
#' heads instead of averaging
#' @param dropout_rate internal dropout rate for attention coefficients
#' @param return_attn_coef if True, return the attention coefficients for
#' the given input (one N x N matrix for each head).
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param attn_kernel_initializer initializer for the attention weights
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param attn_kernel_regularizer regularization applied to the attention kernels
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @param attn_kernel_constraint constraint applied to the attention kernels
#' @export
layer_graph_attention <- function(object,
	channels,
	attn_heads = 1,
	concat_heads = TRUE,
	dropout_rate = 0.5,
	return_attn_coef = FALSE,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	attn_kernel_initializer = 'glorot_uniform',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	attn_kernel_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	attn_kernel_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		attn_heads = as.integer(attn_heads),
		concat_heads = as.integer(concat_heads),
		dropout_rate = dropout_rate,
		return_attn_coef = return_attn_coef,
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		attn_kernel_initializer = attn_kernel_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		attn_kernel_regularizer = attn_kernel_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint,
		attn_kernel_constraint = attn_kernel_constraint
		)
	keras::create_layer(spk$layers$GraphAttention, object, args)
}

#' GraphConv
#' 
#' @description \loadmathjax 
#' A graph convolutional layer (GCN) as presented by
#' [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' This layer computes:
#' \mjdeqn{ Z = \hat D^{-1/2} \hat A \hat D^{-1/2} X W + b }{}
#' where \mjeqn{ \hat A = A + I }{} is the adjacency matrix with added self-loops
#' and \mjeqn{\hat D}{} is its degree matrix.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Modified Laplacian of shape `([batch], N, N)`; can be computed with
#' `spektral.utils.convolution.localpooling_filter`.
#' 
#' **Output**
#' 
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#' 
#' 
#' @param channels number of output channels
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_graph_conv <- function(object,
	channels,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$GraphConv, object, args)
}

#' GraphConvSkip
#' 
#' @description \loadmathjax 
#' A simple convolutional layer with a skip connection.
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' This layer computes:
#' \mjdeqn{ Z = D^{-1/2} A D^{-1/2} X W_1 + X W_2 + b }{}
#' where \mjeqn{ A }{} does not have self-loops (unlike in GraphConv).
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Normalized adjacency matrix of shape `([batch], N, N)`; can be computed
#' with `spektral.utils.convolution.normalized_adjacency`.
#' 
#' **Output**
#' 
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#' 
#' 
#' @param channels number of output channels
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_graph_conv_skip <- function(object,
	channels,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$GraphConvSkip, object, args)
}

#' GraphSageConv
#' 
#' @description \loadmathjax 
#' A GraphSAGE layer as presented by
#' [Hamilton et al. (2017)](https://arxiv.org/abs/1706.02216).
#' 
#' **Mode**: single, disjoint.
#' 
#' This layer computes:
#' \mjdeqn{ Z = [ \textrm{AGGREGATE}(X) \| X ] W + b; \\ Z = \frac{Z}{\|Z\|} }{}
#' where \mjeqn{ \textrm{AGGREGATE} }{} is a function to aggregate a node's
#' neighbourhood. The supported aggregation methods are: sum, mean,
#' max, min, and product.
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`;
#' - Binary adjacency matrix of shape `(N, N)`.
#' 
#' **Output**
#' 
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#' 
#' 
#' @param channels number of output channels
#' @param aggregate_op str, aggregation method to use (`'sum'`, `'mean'`,
#' `'max'`, `'min'`, `'prod'`)
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_graph_sage_conv <- function(object,
	channels,
	aggregate_op = 'mean',
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		aggregate_op = aggregate_op,
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$GraphSageConv, object, args)
}

#' TAGConv
#' 
#' @description \loadmathjax 
#' A Topology Adaptive Graph Convolutional layer (TAG) as presented by
#' [Du et al. (2017)](https://arxiv.org/abs/1710.10370).
#' 
#' **Mode**: single, disjoint.
#' 
#' **This layer expects a sparse adjacency matrix.**
#' 
#' This layer computes:
#' \mjdeqn{ Z = \sum\limits_{k=0}^{K} D^{-1/2}A^kD^{-1/2}XW^{(k)} }{}
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`;
#' - Binary adjacency matrix of shape `(N, N)`.
#' 
#' **Output**
#' 
#' - Node features with the same shape of the input, but the last dimension
#' changed to `channels`.
#' 
#' 
#' @param channels integer, number of output channels
#' @param K the order of the layer (i.e., the layer will consider a K-hop
#' neighbourhood for each node)
#' @param activation activation function to use
#' @param use_bias bool, add a bias vector to the output
#' @param kernel_initializer initializer for the weights
#' @param bias_initializer initializer for the bias vector
#' @param kernel_regularizer regularization applied to the weights
#' @param bias_regularizer regularization applied to the bias vector
#' @param activity_regularizer regularization applied to the output
#' @param kernel_constraint constraint applied to the weights
#' @param bias_constraint constraint applied to the bias vector.
#' @export
layer_tag_conv <- function(object,
	channels,
	K = 3,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	activity_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		K = as.integer(K),
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		activity_regularizer = activity_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$TAGConv, object, args)
}

