#' A graph convolutional layer (GCN)
#'
#' @description \loadmathjax A graph convolutional layer (GCN) as presented by
#' [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907). \cr
#' **Mode**: single, disjoint, mixed, batch. This layer computes:
#' \mjdeqn{\boldsymbol{Z} = \hat D^{-1/2} \hat A \hat D^{-1/2} X W + b}{Z =
#' D_hat^(-1/2)*A_hat*D_hat^(-1/2)*XW + b}
#' where \eqn{\( \hat \A = \A + \I \)}{A_hat = A + I} is the adjacency
#' matrix with added self-loops and \eqn{\(\hat\D\)}{D_hat} is its degree matrix. \cr \cr
#' **Input** \cr
#' A list of: \cr
#' - Node features of shape `([batch], N, F)`;
#' - Modified Laplacian of shape `([batch], N, N)`; can be computed with
#' `spektral.utils.convolution.localpooling_filter`. \cr \cr
#' **Output** \cr
#' - Node features with the same shape as the input, but with the last
#' dimension changed to `channels`.
#'
#' See the online documentation for [Spektral (python package)](https://graphneural.network/layers/convolution/#graphconv)
#' for more details.
#' @author Daniele Grattarola (python code); Russell Dinnage (R wrapper)
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
#'
#' A Chebyshev convolutional layer as presented by
#' [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375). \cr
#' **Mode**: single, disjoint, mixed, batch. This layer computes:
#' \deqn{\Z = \sum \limits_{k=0}^{K - 1} \T^{(k)} \W^{(k)} + \b^{(k)}}{Z =
#' sum_{k=0}^{K - 1}(T^((k))*W^((k)) + b^((k)))}
#' where \eqn{ \T^{(0)}, ..., \T^{(K - 1)}}{T^((0)), ..., T^((K - 1))} are Chebyshev
#' polynomials of \eqn{L_tilde} defined as
#' \deqn{ \T^{(0)} = \X \\ \T^{(1)} = \tilde \L \X \\ \T^{(k \ge 2)} = 2 \cdot \tilde \L \T^{(k - 1)} - \T^{(k - 2)}}{T^((0)) =
#' X; T^((1)) = L_tilde*X;  T^((k > 2)) = 2 * L_tilde T^((k - 1)) - T^((k - 2))}
#' where \deqn{ \tilde \L = \frac{2}{\lambda_{max}} \cdot (\I - \D^{-1/2} \A \D^{-1/2}) - \I}{L_tilde =
#' (2 / lambda_max) * (I - D^(-1/2) * A * D^(-1/2)) - I}
#' is the normalized Laplacian with a rescaled spectrum. \cr \cr
#' **Input** \cr
#' - Node features of shape `([batch], N, F)`;
#' - A list of K Chebyshev polynomials of shape
#' `[([batch], N, N), ..., ([batch], N, N)]`; can be computed with
#' `spektral.utils.convolution.chebyshev_filter`. \cr \cr
#' **Output** \cr
#' - Node features with the same shape of the input, but with the last
#' dimension changed to `channels`.
#'
#' See the online documentation for [Spektral (python package)](https://graphneural.network/layers/convolution/#chebconv)
#' for more details.
#' @author Daniele Grattarola (python code); Russell Dinnage (R wrapper)
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

#' A graph convolutional layer with ARMA_K filters.
#'
#' A graph convolutional layer with ARMA_K filters, as presented by
#' [Bianchi et al. (2019)](https://arxiv.org/abs/1901.01343). \cr
#' **Mode**: single, disjoint, mixed, batch. This layer computes:
#' \deqn{Z = \frac{1}{K} \sum\limits_{k=1}^K \bar X_k^{(T)}}{Z =
#' (1/K) sum_{k=1}^K(Xbar_k^((T))}
#' where \eqn{K} is the order of the ARMA_K filter, and where:
#' \deqn{ \bar \X_k^{(t + 1)} = \sigma \left(\tilde \L \bar \X^{(t)} \W^{(t)} + \X \V^{(t)} \right)}{Xbar_k^((t + 1)) =
#' sigma *(L_tilde Xbar^((t))*W^((t)) + X*V^((t)))}
#' is a recursive approximation of an ARMA_1 filter, where
#' \eqn{\bar X^{(0)} = X}{Xbar^((0)) = X}
#' and
#' \deqn{\tilde L = \frac{2}{\lambda_{max}} \cdot (I - D^{-1/2} A D^{-1/2}) - I}{L_tilde =
#' (2/lambda_max) * (I - D^(-1/2) * A * D^(-1/2)) - I}
#' is the normalized Laplacian with a rescaled spectrum. \cr \cr
#' **Input**
#' - Node features of shape `([batch], N, F)`;
#' - Normalized and rescaled Laplacian of shape `([batch], N, N)`; can be
#' computed with `spektral.utils.convolution.normalized_laplacian` and
#' `spektral.utils.convolution.rescale_laplacian`. \cr \cr
#' **Output**
#' - Node features with the same shape as the input, but with the last
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
                            attn_heads              = 1,
                            concat_heads            = TRUE,
                            dropout_rate            = 0.5,
                            return_attn_coef        = FALSE,
                            activation              = NULL,
                            use_bias                = TRUE,
                            kernel_initializer      = 'glorot_uniform',
                            bias_initializer        = 'zeros',
                            attn_kernel_initializer = 'glorot_uniform',
                            kernel_regularizer      = NULL,
                            bias_regularizer        = NULL,
                            attn_kernel_regularizer = NULL,
                            activity_regularizer    = NULL,
                            kernel_constraint       = NULL,
                            bias_constraint         = NULL,
                            attn_kernel_constraint  = NULL,
                            ...) {

  args <- list(channels             = as.integer(channels),
               attn_heads           = as.integer(attn_heads),
               concat_heads         = as.integer(concat_heads),
               return_attn_coef     = return_attn_coef,
               activation           = activation,
               dropout_rate         = dropout_rate,
               activation           = activation,
               use_bias             = use_bias,
               kernel_initializer   = kernel_initializer,
               bias_initializer     = bias_initializer,
               attn_kernel_initializer   = attn_kernel_initializer,
               kernel_regularizer   = kernel_regularizer,
               bias_regularizer     = bias_regularizer,
               attn_kernel_regularizer   = attn_kernel_regularizer,
               activity_regularizer = activity_regularizer,
               kernel_constraint    = kernel_constraint,
               bias_constraint      = bias_constraint,
               attn_kernel_constraint    = attn_kernel_constraint,
               ...)

  keras::create_layer(spk$layers$GraphAttention, object, args)

}
