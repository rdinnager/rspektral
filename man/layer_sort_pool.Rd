% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layers_pool.R
\name{layer_sort_pool}
\alias{layer_sort_pool}
\title{SortPool}
\usage{
layer_sort_pool(object, k, ...)
}
\arguments{
\item{k}{integer, number of nodes to keep}
}
\description{
\loadmathjax
A SortPool layer as described by
\href{https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf}{Zhang et al}.
This layers takes a graph signal \mjeqn{\mathbf{X}}{} and returns the topmost k
rows according to the last column.
If \mjeqn{\mathbf{X}}{} has less than k rows, the result is zero-padded to k.

\strong{Mode}: single, disjoint, batch.

\strong{Input}
\itemize{
\item Node features of shape \verb{([batch], N, F)};
\item Graph IDs of shape \verb{(N, )} (only in disjoint mode);
}

\strong{Output}
\itemize{
\item Pooled node features of shape \verb{(batch, k, F)} (if single mode, shape will
be \verb{(1, k, F)}).
}
}
