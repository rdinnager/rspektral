% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layers_conv.R
\name{layer_graph_sage_conv}
\alias{layer_graph_sage_conv}
\title{GraphSageConv}
\usage{
layer_graph_sage_conv(
  object,
  channels,
  aggregate_op = "mean",
  activation = NULL,
  use_bias = TRUE,
  kernel_initializer = "glorot_uniform",
  bias_initializer = "zeros",
  kernel_regularizer = NULL,
  bias_regularizer = NULL,
  activity_regularizer = NULL,
  kernel_constraint = NULL,
  bias_constraint = NULL,
  ...
)
}
\arguments{
\item{channels}{number of output channels}

\item{aggregate_op}{str, aggregation method to use (\code{'sum'}, \code{'mean'},
\code{'max'}, \code{'min'}, \code{'prod'})}

\item{activation}{activation function to use}

\item{use_bias}{bool, add a bias vector to the output}

\item{kernel_initializer}{initializer for the weights}

\item{bias_initializer}{initializer for the bias vector}

\item{kernel_regularizer}{regularization applied to the weights}

\item{bias_regularizer}{regularization applied to the bias vector}

\item{activity_regularizer}{regularization applied to the output}

\item{kernel_constraint}{constraint applied to the weights}

\item{bias_constraint}{constraint applied to the bias vector.}
}
\description{
\loadmathjax
A GraphSAGE layer as presented by
\href{https://arxiv.org/abs/1706.02216}{Hamilton et al. (2017)}.

\strong{Mode}: single, disjoint.

This layer computes:
\mjdeqn{ Z = \link[= \\textrm{AGGREGATE}(X) \\| X ]{ \textrm{AGGREGATE}(X) \| X } W + b; \\ Z = \frac{Z}{\|Z\|} }{}
where \mjeqn{ \textrm{AGGREGATE} }{} is a function to aggregate a node's
neighbourhood. The supported aggregation methods are: sum, mean,
max, min, and product.

\strong{Input}
\itemize{
\item Node features of shape \verb{(N, F)};
\item Binary adjacency matrix of shape \verb{(N, N)}.
}

\strong{Output}
\itemize{
\item Node features with the same shape as the input, but with the last
dimension changed to \code{channels}.
}
}
