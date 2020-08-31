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

#' A Chebyshev graph convolutional layer.
#' A Chebyshev convolutional layer as presented by
#' [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375). \cr
#' **Mode**: single, disjoint, mixed, batch. This layer computes:
#' $$ \Z = \sum \limits_{k=0}^{K - 1} \T^{(k)} \W^{(k)} + \b^{(k)},
#' $$
#' where \( \T^{(0)}, ..., \T^{(K - 1)} \) are Chebyshev polynomials of \(\tilde \L\)
#' defined as
#' $$ \T^{(0)} = \X \\ \T^{(1)} = \tilde \L \X \\ \T^{(k \ge 2)} = 2 \cdot \tilde \L \T^{(k - 1)} - \T^{(k - 2)},
#' $$
#' where
#' $$ \tilde \L = \frac{2}{\lambda_{max}} \cdot (\I - \D^{-1/2} \A \D^{-1/2}) - \I
#' $$
#' is the normalized Laplacian with a rescaled spectrum. **Input** - Node features of shape `([batch], N, F)`;
#' - A list of K Chebyshev polynomials of shape
#' `[([batch], N, N), ..., ([batch], N, N)]`; can be computed with
#' `spektral.utils.convolution.chebyshev_filter`. **Output** - Node features with the same shape of the input, but with the last
#' dimension changed to `channels`.
#' @param object model or layer object;
#' @param channels number of output channels;
#' @param K order of the Chebyshev polynomials;
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
layer_cheb_conv <- function(object,
                            channels,
                            K                    = 1L,
                            activation           = NULL,
                            use_bias             = TRUE,
                            kernel_initializer   = 'glorot_uniform',
                            bias_initializer     = 'zeros',
                            kernel_regularizer   = NULL,
                            bias_regularizer     = NULL,
                            activity_regularizer = NULL,
                            kernel_constraint    = NULL,
                            bias_constraint      = NULL,
                            ...) {

  args <- list(channels             = as.integer(channels),
               K                    = as.integer(K),
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

  keras::create_layer(spk$layers$ChebConv, object, args)

}

#' A graph convolutional layer with ARMA\(_K\) filters.
#'
#' A graph convolutional layer with ARMA\(_K\) filters, as presented by
#' [Bianchi et al. (2019)](https://arxiv.org/abs/1901.01343). **Mode**: single, disjoint, mixed, batch. This layer computes:
#' $$ \Z = \frac{1}{K} \sum\limits_{k=1}^K \bar\X_k^{(T)},
#' $$
#' where \(K\) is the order of the ARMA\(_K\) filter, and where:
#' $$ \bar \X_k^{(t + 1)} = \sigma \left(\tilde \L \bar \X^{(t)} \W^{(t)} + \X \V^{(t)} \right)
#' $$
#' is a recursive approximation of an ARMA\(_1\) filter, where
#' \( \bar \X^{(0)} = \X \)
#' and
#' $$ \tilde \L = \frac{2}{\lambda_{max}} \cdot (\I - \D^{-1/2} \A \D^{-1/2}) - \I
#' $$
#' is the normalized Laplacian with a rescaled spectrum. **Input** - Node features of shape `([batch], N, F)`;
#' - Normalized and rescaled Laplacian of shape `([batch], N, N)`; can be
#' computed with `spektral.utils.convolution.normalized_laplacian` and
#' `spektral.utils.convolution.rescale_laplacian`. **Output** - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#' @param object model or layer object;
#' @param channels number of output channels;
#' @param order  order of the full ARMA\(_K\) filter, i.e., the number of parallel
#' stacks in the layer;
#' @param iterations number of iterations to compute each ARMA\(_1\) approximation;
#' @param share_weights share the weights in each ARMA\(_1\) stack.
#' @param gcn_activation activation function to use to compute each ARMA\(_1\)
#' stack;
#' @param dropout_rate dropout rate for skip connection;
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
layer_arma_conv <- function(object,
                            channels,
                            order                = 1,
                            iterations           = 1,
                            share_weights        = FALSE,
                            gcn_activation       = 'relu',
                            dropout_rate         = 0.0,
                            activation           = NULL,
                            use_bias             = TRUE,
                            kernel_initializer   = 'glorot_uniform',
                            bias_initializer     = 'zeros',
                            kernel_regularizer   = NULL,
                            bias_regularizer     = NULL,
                            activity_regularizer = NULL,
                            kernel_constraint    = NULL,
                            bias_constraint      = NULL,
                            ...) {

  args <- list(channels             = as.integer(channels),
               order                = as.integer(K),
               iterations           = as.integer(iterations),
               share_weights        = share_weights,
               gcn_activation       = gcn_activation,
               dropout_rate         = dropout_rate,
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

  keras::create_layer(spk$layers$ARMAConv, object, args)

}
