#' A graph convolutional layer (GCN)
#'
#' A graph convolutional layer (GCN) as presented by
#' [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907). \cr
#' **Mode**: single, disjoint, mixed, batch. This layer computes:
#' \deqn{\Z = \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \X \W + \b}
#' where \eqn{\( \hat \A = \A + \I \)} is the adjacency matrix with added self-loops
#' and \eqn{\(\hat\D\)} is its degree matrix. \cr \cr
#' **Input** \cr
#' A list of: \cr
#' - Node features of shape `([batch], N, F)`;
#' - Modified Laplacian of shape `([batch], N, N)`; can be computed with
#' `spektral.utils.convolution.localpooling_filter`. \cr \cr
#' **Output** \cr
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#' @param object model or layer object;
#' @param channels number of output channels;
#' @param activation activation function to use;
#' @param use_bias bool, add a bias vector to the output;
#' @param kernel_initializer initializer for the weights;
#' @param bias_initializer initializer for the bias vector;
#' @param kernel_regularizer regularization applied to the weights;
#' @param bias_regularizer regularization applied to the bias vector;
#' @param activity_regularizer regularization applied to the output;
#' @param kernel_constraint constraint applied to the weights;
#' @param bias_constraint constraint applied to the bias vector.
#'
#' @export
layer_graph_conv <- function(object,
                             channels,
                             activation           = None,
                             use_bias             = True,
                             kernel_initializer   = 'glorot_uniform',
                             bias_initializer     = 'zeros',
                             kernel_regularizer   = NULL,
                             bias_regularizer     = NULL,
                             activity_regularizer = NULL,
                             kernel_constraint    = NULL,
                             bias_constraint      = NULL,
                             ...) {

  args <- list(channels             = as.integer(channels),
               activation           = activation,
               use_bias             = use_bias,
               kernel_initializer   = kernel_initializer,
               bias_initializer     = bias_initializer,
               kernel_regularizer   = kernel_regularizer,
               bias_regularizer     = bias_regularizer,
               activity_regularizer = activity_regularizer,
               kernel_constraint    = kernel_constraint,
               bias_constraint      = bias_constraint,
               ...)

  keras::create_layer(spk$layers$GraphConv, object, args)

}

#' Calculate modified Laplacian on an adjacency matrix
#'
#' @param A Adjacency matrix as a sparse matrix (\code{\link[Matrix]{dgRMatrix}})
#' @param densify Should the Laplacian be returned as a dense matrix (default is sparse)?
#'
#' @return The Laplacian as a sparse matrix (or a dense matrix if \code{densify} is \code{TRUE})
#' @export
preprocess_laplacian_mod <- function(A, densify = FALSE) {
  res <- spk$layers$GraphConv$preprocess(A)
  if(densify) {
    res <- as.matrix(res)
  }

  res

}
