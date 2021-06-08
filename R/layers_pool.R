#' DiffPool
#' 
#' @description \loadmathjax 
#' A DiffPool layer as presented by
#' [Ying et al. (2018)](https://arxiv.org/abs/1806.08804).
#' 
#' **Mode**: batch.
#' 
#' This layer computes a soft clustering \mjeqn{\boldsymbol{S}}{} of the input graphs using a GNN,
#' and reduces graphs as follows:
#' 
#' \mjdeqn{\boldsymbol{S} = \textrm{GNN}(\boldsymbol{A}, \boldsymbol{X}); \\\boldsymbol{A}' = \boldsymbol{S}^\top \boldsymbol{A} \boldsymbol{S}; \boldsymbol{X}' = \boldsymbol{S}^\top \boldsymbol{X};}{}
#' 
#' where GNN consists of one GraphConv layer with softmax activation.
#' Two auxiliary loss terms are also added to the model: the _link prediction
#' loss_
#' \mjdeqn{\big\| \boldsymbol{A} - \boldsymbol{S}\S^\top \big\| _ F}{}
#' and the _entropy loss_
#' \mjdeqn{- \frac{1}{N} \sum\limits_{i = 1}^{N} \boldsymbol{S} \log (\boldsymbol{S}).}{}
#' 
#' The layer also applies a 1-layer GCN to the input features, and returns
#' the updated graph signal (the number of output channels is controlled by
#' the `channels` parameter).
#' The layer can be used without a supervised loss, to compute node clustering
#' simply by minimizing the two auxiliary losses.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Binary adjacency matrix of shape `([batch], N, N)`;
#' 
#' **Output**
#' 
#' - Reduced node features of shape `([batch], K, channels)`;
#' - Reduced adjacency matrix of shape `([batch], K, K)`;
#' - If `return_mask=True`, the soft clustering matrix of shape `([batch], N, K)`.
#' 
#' 

#' @export
layer_diff_pool <- function(object,
	k,
	channels = NULL,
	return_mask = FALSE,
	activation = NULL,
	kernel_initializer = 'glorot_uniform',
	kernel_regularizer = NULL,
	kernel_constraint = NULL,
	...)
{
	args <- list(k = as.integer(k),
		channels = as.integer(channels),
		return_mask = return_mask,
		activation = activation,
		kernel_initializer = kernel_initializer,
		kernel_regularizer = kernel_regularizer,
		kernel_constraint = kernel_constraint
		)
	keras::create_layer(spk$layers$DiffPool, object, args)
}

#' GlobalAttentionPool
#' 
#' @description \loadmathjax 
#' A gated attention global pooling layer as presented by
#' [Li et al. (2017)](https://arxiv.org/abs/1511.05493).
#' 
#' This layer computes:
#' \mjdeqn{\boldsymbol{X}' = \sum\limits_{i=1}^{N} (\sigma(\boldsymbol{X} \boldsymbol{W} _ 1 + \boldsymbol{b} _ 1) \odot (\boldsymbol{X} \boldsymbol{W} _ 2 + \boldsymbol{b} _ 2))_i}{}
#' where \mjeqn{\sigma}{} is the sigmoid activation function.
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Graph IDs of shape `(N, )` (only in disjoint mode);
#' 
#' **Output**
#' 
#' - Pooled node features of shape `(batch, channels)` (if single mode,
#' shape will be `(1, channels)`).
#' 
#' 
#' @param channels integer, number of output channels
#' @param kernel_initializer NA
#' @param bias_initializer initializer for the bias vectors
#' @param kernel_regularizer regularization applied to the kernel matrices
#' @param bias_regularizer regularization applied to the bias vectors
#' @param kernel_constraint constraint applied to the kernel matrices
#' @param bias_constraint constraint applied to the bias vectors.
#' @export
layer_global_attention_pool <- function(object,
	channels,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(channels = as.integer(channels),
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$GlobalAttentionPool, object, args)
}

#' GlobalAttnSumPool
#' 
#' @description \loadmathjax 
#' A node-attention global pooling layer. Pools a graph by learning attention
#' coefficients to sum node features.
#' 
#' This layer computes:
#' \mjdeqn{\alpha = \textrm{softmax}( \boldsymbol{X} \boldsymbol{a}); \\\boldsymbol{X}' = \sum\limits_{i=1}^{N} \alpha_i \cdot \boldsymbol{X} _ i}{}
#' where \mjeqn{\boldsymbol{a} \in \mathbb{R}^F}{} is a trainable vector. Note that the softmax
#' is applied across nodes, and not across features.
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Graph IDs of shape `(N, )` (only in disjoint mode);
#' 
#' **Output**
#' 
#' - Pooled node features of shape `(batch, F)` (if single mode, shape will
#' be `(1, F)`).
#' 
#' 
#' @param attn_kernel_initializer initializer for the attention weights
#' @param attn_kernel_regularizer regularization applied to the attention kernel
#' matrix
#' @param attn_kernel_constraint constraint applied to the attention kernel
#' matrix
#' @export
layer_global_attn_sum_pool <- function(object,
	attn_kernel_initializer = 'glorot_uniform',
	attn_kernel_regularizer = NULL,
	attn_kernel_constraint = NULL,
	...)
{
	args <- list(attn_kernel_initializer = attn_kernel_initializer,
		attn_kernel_regularizer = attn_kernel_regularizer,
		attn_kernel_constraint = attn_kernel_constraint
		)
	keras::create_layer(spk$layers$GlobalAttnSumPool, object, args)
}

#' GlobalAvgPool
#' 
#' @description \loadmathjax 
#' An average pooling layer. Pools a graph by computing the average of its node
#' features.
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Graph IDs of shape `(N, )` (only in disjoint mode);
#' 
#' **Output**
#' 
#' - Pooled node features of shape `(batch, F)` (if single mode, shape will
#' be `(1, F)`).
#' 
#' **Arguments**
#' 
#' None.
#' 
#' 

#' @export
layer_global_avg_pool <- function(object,
		...)
{
	args <- list(
		)
	keras::create_layer(spk$layers$GlobalAvgPool, object, args)
}

#' GlobalMaxPool
#' 
#' @description \loadmathjax 
#' A max pooling layer. Pools a graph by computing the maximum of its node
#' features.
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Graph IDs of shape `(N, )` (only in disjoint mode);
#' 
#' **Output**
#' 
#' - Pooled node features of shape `(batch, F)` (if single mode, shape will
#' be `(1, F)`).
#' 
#' **Arguments**
#' 
#' None.
#' 
#' 

#' @export
layer_global_max_pool <- function(object,
		...)
{
	args <- list(
		)
	keras::create_layer(spk$layers$GlobalMaxPool, object, args)
}

#' GlobalPooling
#' 
#' @description \loadmathjax NA

#' @export
layer_global_pooling <- function(object,
		...)
{
	args <- list(
		)
	keras::create_layer(spk$layers$GlobalPooling, object, args)
}

#' GlobalSumPool
#' 
#' @description \loadmathjax 
#' A global sum pooling layer. Pools a graph by computing the sum of its node
#' features.
#' 
#' **Mode**: single, disjoint, mixed, batch.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Graph IDs of shape `(N, )` (only in disjoint mode);
#' 
#' **Output**
#' 
#' - Pooled node features of shape `(batch, F)` (if single mode, shape will
#' be `(1, F)`).
#' 
#' **Arguments**
#' 
#' None.
#' 
#' 

#' @export
layer_global_sum_pool <- function(object,
		...)
{
	args <- list(
		)
	keras::create_layer(spk$layers$GlobalSumPool, object, args)
}

#' MinCutPool
#' 
#' @description \loadmathjax 
#' A minCUT pooling layer as presented by
#' [Bianchi et al. (2019)](https://arxiv.org/abs/1907.00481).
#' 
#' **Mode**: batch.
#' 
#' This layer computes a soft clustering \mjeqn{\boldsymbol{S}}{} of the input graphs using a MLP,
#' and reduces graphs as follows:
#' 
#' \mjdeqn{\boldsymbol{S} = \textrm{MLP}(\boldsymbol{X}); \\\boldsymbol{A}' = \boldsymbol{S}^\top \boldsymbol{A} \boldsymbol{S}; \boldsymbol{X}' = \boldsymbol{S}^\top \boldsymbol{X};}{}
#' 
#' where MLP is a multi-layer perceptron with softmax output.
#' Two auxiliary loss terms are also added to the model: the _minCUT loss_
#' \mjdeqn{- \frac{ \mathrm{Tr}(\boldsymbol{S}^\top \boldsymbol{A} \boldsymbol{S}) }{ \mathrm{Tr}(\boldsymbol{S}^\top \boldsymbol{D} \boldsymbol{S}) }}{}
#' and the _orthogonality loss_
#' \mjdeqn{\left\|\frac{\boldsymbol{S}^\top \boldsymbol{S}}{\| \boldsymbol{S}^\top \boldsymbol{S} \| _ F}- \frac{\boldsymbol{I} _ K}{\sqrt{K}}\right\| _ F.}{}
#' 
#' The layer can be used without a supervised loss, to compute node clustering
#' simply by minimizing the two auxiliary losses.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Binary adjacency matrix of shape `([batch], N, N)`;
#' 
#' **Output**
#' 
#' - Reduced node features of shape `([batch], K, F)`;
#' - Reduced adjacency matrix of shape `([batch], K, K)`;
#' - If `return_mask=True`, the soft clustering matrix of shape `([batch], N, K)`.
#' 
#' 

#' @export
layer_min_cut_pool <- function(object,
	k,
	mlp_hidden = NULL,
	mlp_activation = 'relu',
	return_mask = FALSE,
	activation = NULL,
	use_bias = TRUE,
	kernel_initializer = 'glorot_uniform',
	bias_initializer = 'zeros',
	kernel_regularizer = NULL,
	bias_regularizer = NULL,
	kernel_constraint = NULL,
	bias_constraint = NULL,
	...)
{
	args <- list(k = as.integer(k),
		mlp_hidden = mlp_hidden,
		mlp_activation = mlp_activation,
		return_mask = return_mask,
		activation = activation,
		use_bias = use_bias,
		kernel_initializer = kernel_initializer,
		bias_initializer = bias_initializer,
		kernel_regularizer = kernel_regularizer,
		bias_regularizer = bias_regularizer,
		kernel_constraint = kernel_constraint,
		bias_constraint = bias_constraint
		)
	keras::create_layer(spk$layers$MinCutPool, object, args)
}

#' SAGPool
#' 
#' @description \loadmathjax 
#' A self-attention graph pooling layer as presented by
#' [Lee et al. (2019)](https://arxiv.org/abs/1904.08082).
#' 
#' **Mode**: single, disjoint.
#' 
#' This layer computes the following operations:
#' 
#' \mjdeqn{\boldsymbol{y} = \textrm{GNN}(\boldsymbol{A}, \boldsymbol{X}); \;\;\;\;\boldsymbol{i} = \textrm{rank}(\boldsymbol{y}, K); \;\;\;\;\boldsymbol{X}' = (\boldsymbol{X} \odot \textrm{tanh}(\boldsymbol{y}))_\boldsymbol{i}; \;\;\;\;\boldsymbol{A}' = \boldsymbol{A} _ {\boldsymbol{i}, \boldsymbol{i}}}{}
#' 
#' where \mjeqn{ \textrm{rank}(\boldsymbol{y}, K) }{} returns the indices of the top K values of
#' \mjeqn{\boldsymbol{y}}{}, and \mjeqn{\textrm{GNN}}{} consists of one GraphConv layer with no
#' activation. \mjeqn{K}{} is defined for each graph as a fraction of the number of
#' nodes.
#' 
#' This layer temporarily makes the adjacency matrix dense in order to compute
#' \mjeqn{\boldsymbol{A}'}{}.
#' If memory is not an issue, considerable speedups can be achieved by using
#' dense graphs directly.
#' Converting a graph from sparse to dense and back to sparse is an expensive
#' operation.
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`;
#' - Binary adjacency matrix of shape `(N, N)`;
#' - Graph IDs of shape `(N, )` (only in disjoint mode);
#' 
#' **Output**
#' 
#' - Reduced node features of shape `(ratio * N, F)`;
#' - Reduced adjacency matrix of shape `(ratio * N, ratio * N)`;
#' - Reduced graph IDs of shape `(ratio * N, )` (only in disjoint mode);
#' - If `return_mask=True`, the binary pooling mask of shape `(ratio * N, )`.
#' 
#' 
#' @param ratio float between 0 and 1, ratio of nodes to keep in each graph
#' @param return_mask boolean, whether to return the binary mask used for pooling
#' @param sigmoid_gating boolean, use a sigmoid gating activation instead of a
#' tanh
#' @param kernel_initializer initializer for the weights
#' @param kernel_regularizer regularization applied to the weights
#' @param kernel_constraint constraint applied to the weights
#' @export
layer_sag_pool <- function(object,
	ratio,
	return_mask = FALSE,
	sigmoid_gating = FALSE,
	kernel_initializer = 'glorot_uniform',
	kernel_regularizer = NULL,
	kernel_constraint = NULL,
	...)
{
	args <- list(ratio = ratio,
		return_mask = return_mask,
		sigmoid_gating = sigmoid_gating,
		kernel_initializer = kernel_initializer,
		kernel_regularizer = kernel_regularizer,
		kernel_constraint = kernel_constraint
		)
	keras::create_layer(spk$layers$SAGPool, object, args)
}

#' SortPool
#' 
#' @description \loadmathjax 
#' A SortPool layer as described by
#' [Zhang et al](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf).
#' This layers takes a graph signal \mjeqn{\mathbf{X}}{} and returns the topmost k
#' rows according to the last column.
#' If \mjeqn{\mathbf{X}}{} has less than k rows, the result is zero-padded to k.
#' 
#' **Mode**: single, disjoint, batch.
#' 
#' **Input**
#' 
#' - Node features of shape `([batch], N, F)`;
#' - Graph IDs of shape `(N, )` (only in disjoint mode);
#' 
#' **Output**
#' 
#' - Pooled node features of shape `(batch, k, F)` (if single mode, shape will
#' be `(1, k, F)`).
#' 
#' 
#' @param k integer, number of nodes to keep
#' @export
layer_sort_pool <- function(object,
	k,
	...)
{
	args <- list(k = as.integer(k)
		)
	keras::create_layer(spk$layers$SortPool, object, args)
}

#' TopKPool
#' 
#' @description \loadmathjax 
#' A gPool/Top-K layer as presented by
#' [Gao & Ji (2019)](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf) and
#' [Cangea et al. (2018)](https://arxiv.org/abs/1811.01287).
#' 
#' **Mode**: single, disjoint.
#' 
#' This layer computes the following operations:
#' 
#' \mjdeqn{\boldsymbol{y} = \frac{\boldsymbol{X}\p}{\|\boldsymbol{p}\|}; \;\;\;\;\boldsymbol{i} = \textrm{rank}(\boldsymbol{y}, K); \;\;\;\;\boldsymbol{X}' = (\boldsymbol{X} \odot \textrm{tanh}(\boldsymbol{y}))_\boldsymbol{i}; \;\;\;\;\boldsymbol{A}' = \boldsymbol{A} _ {\boldsymbol{i}, \boldsymbol{i}}}{}
#' 
#' where \mjeqn{ \textrm{rank}(\boldsymbol{y}, K) }{} returns the indices of the top K values of
#' \mjeqn{\boldsymbol{y}}{}, and \mjeqn{\boldsymbol{p}}{} is a learnable parameter vector of size \mjeqn{F}{}. \mjeqn{K}{} is
#' defined for each graph as a fraction of the number of nodes.
#' Note that the the gating operation \mjeqn{\textrm{tanh}(\boldsymbol{y})}{} (Cangea et al.)
#' can be replaced with a sigmoid (Gao & Ji).
#' 
#' This layer temporarily makes the adjacency matrix dense in order to compute
#' \mjeqn{\boldsymbol{A}'}{}.
#' If memory is not an issue, considerable speedups can be achieved by using
#' dense graphs directly.
#' Converting a graph from sparse to dense and back to sparse is an expensive
#' operation.
#' 
#' **Input**
#' 
#' - Node features of shape `(N, F)`;
#' - Binary adjacency matrix of shape `(N, N)`;
#' - Graph IDs of shape `(N, )` (only in disjoint mode);
#' 
#' **Output**
#' 
#' - Reduced node features of shape `(ratio * N, F)`;
#' - Reduced adjacency matrix of shape `(ratio * N, ratio * N)`;
#' - Reduced graph IDs of shape `(ratio * N, )` (only in disjoint mode);
#' - If `return_mask=True`, the binary pooling mask of shape `(ratio * N, )`.
#' 
#' 
#' @param ratio float between 0 and 1, ratio of nodes to keep in each graph
#' @param return_mask boolean, whether to return the binary mask used for pooling
#' @param sigmoid_gating boolean, use a sigmoid gating activation instead of a
#' tanh
#' @param kernel_initializer initializer for the weights
#' @param kernel_regularizer regularization applied to the weights
#' @param kernel_constraint constraint applied to the weights
#' @export
layer_top_k_pool <- function(object,
	ratio,
	return_mask = FALSE,
	sigmoid_gating = FALSE,
	kernel_initializer = 'glorot_uniform',
	kernel_regularizer = NULL,
	kernel_constraint = NULL,
	...)
{
	args <- list(ratio = ratio,
		return_mask = return_mask,
		sigmoid_gating = sigmoid_gating,
		kernel_initializer = kernel_initializer,
		kernel_regularizer = kernel_regularizer,
		kernel_constraint = kernel_constraint
		)
	keras::create_layer(spk$layers$TopKPool, object, args)
}

