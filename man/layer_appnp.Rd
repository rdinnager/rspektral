% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layers_conv.R
\name{layer_appnp}
\alias{layer_appnp}
\title{APPNP}
\usage{
layer_appnp(
  object,
  channels,
  alpha = 0.2,
  propagations = 1,
  mlp_hidden = NULL,
  mlp_activation = "relu",
  dropout_rate = 0,
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

\item{alpha}{teleport probability during propagation}

\item{propagations}{number of propagation steps}

\item{mlp_hidden}{list of integers, number of hidden units for each hidden
layer in the MLP (if None, the MLP has only the output layer)}

\item{mlp_activation}{activation for the MLP layers}

\item{dropout_rate}{dropout rate for Laplacian and MLP layers}

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
A graph convolutional layer implementing the APPNP operator, as presented by
\href{https://arxiv.org/abs/1810.05997}{Klicpera et al. (2019)}.

This layer computes:
\mjdeqn{ Z^{(0)} = \textrm{MLP}(X); \\ Z^{(K)} = (1 - \alpha) \hat D^{-1/2} \hat A \hat D^{-1/2} Z^{(K - 1)} + \alpha Z^{(0)}, }{}
where \mjeqn{\alpha}{} is the \emph{teleport} probability and \mjeqn{\textrm{MLP}}{} is a
multi-layer perceptron.

\strong{Mode}: single, disjoint, mixed, batch.

\strong{Input}
\itemize{
\item Node features of shape \verb{([batch], N, F)};
\item Modified Laplacian of shape \verb{([batch], N, N)}; can be computed with
\code{spektral.utils.convolution.localpooling_filter}.
}

\strong{Output}
\itemize{
\item Node features with the same shape as the input, but with the last
dimension changed to \code{channels}.
}
}
