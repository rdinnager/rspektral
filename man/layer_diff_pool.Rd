% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layers_pool.R
\name{layer_diff_pool}
\alias{layer_diff_pool}
\title{DiffPool}
\usage{
layer_diff_pool(
  object,
  k,
  channels = NULL,
  return_mask = FALSE,
  activation = NULL,
  kernel_initializer = "glorot_uniform",
  kernel_regularizer = NULL,
  kernel_constraint = NULL,
  ...
)
}
\description{
\loadmathjax
A DiffPool layer as presented by
\href{https://arxiv.org/abs/1806.08804}{Ying et al. (2018)}.

\strong{Mode}: batch.

This layer computes a soft clustering \mjeqn{\boldsymbol{S}}{} of the input graphs using a GNN,
and reduces graphs as follows:

\mjdeqn{\boldsymbol{S} = \textrm{GNN}(\boldsymbol{A}, \boldsymbol{X}); \\\boldsymbol{A}' = \boldsymbol{S}^\top \boldsymbol{A} \boldsymbol{S}; \boldsymbol{X}' = \boldsymbol{S}^\top \boldsymbol{X};}{}

where GNN consists of one GraphConv layer with softmax activation.
Two auxiliary loss terms are also added to the model: the \emph{link prediction
loss}
\mjdeqn{\big\| \boldsymbol{A} - \boldsymbol{S}\S^\top \big\| _ F}{}
and the \emph{entropy loss}
\mjdeqn{- \frac{1}{N} \sum\limits_{i = 1}^{N} \boldsymbol{S} \log (\boldsymbol{S}).}{}

The layer also applies a 1-layer GCN to the input features, and returns
the updated graph signal (the number of output channels is controlled by
the \code{channels} parameter).
The layer can be used without a supervised loss, to compute node clustering
simply by minimizing the two auxiliary losses.

\strong{Input}
\itemize{
\item Node features of shape \verb{([batch], N, F)};
\item Binary adjacency matrix of shape \verb{([batch], N, N)};
}

\strong{Output}
\itemize{
\item Reduced node features of shape \verb{([batch], K, channels)};
\item Reduced adjacency matrix of shape \verb{([batch], K, K)};
\item If \code{return_mask=True}, the soft clustering matrix of shape \verb{([batch], N, K)}.
}
}
