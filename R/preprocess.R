#' Preprocess Adjacency Matrix for use with layer_arma_conv
#'
#'This utility function can be used to  preprocess a network adjacency matrix
#'into an object that can be used to represent the network in the \code{\link{layer_arma_conv}} layer.
#'Internally it does this:\cr
#'	\code{fltr=\link{utils_normalized_laplacian}(A,symmetric=True)}\cr
#'	\code{fltr=\link{utils_rescale_laplacian}(fltr,lmax=2)}\cr
#'	\code{fltr}\cr
#'
#'@param A An Adjacency matrix (can be dense or sparse)
#'
#'@export
preprocess_arma_conv <- function(A) {
	spk$layers$ARMAConv$preprocess(A)
}

#' Preprocess Adjacency Matrix for use with layer_cheb_conv
#'
#'This utility function can be used to  preprocess a network adjacency matrix
#'into an object that can be used to represent the network in the \code{\link{layer_cheb_conv}} layer.
#'Internally it does this:\cr
#'	\code{L=\link{utils_normalized_laplacian}(A)}\cr
#'	\code{L=\link{utils_rescale_laplacian}(L)}\cr
#'	\code{L}\cr
#'
#'@param A An Adjacency matrix (can be dense or sparse)
#'
#'@export
preprocess_cheb_conv <- function(A) {
	spk$layers$ChebConv$preprocess(A)
}

#' Preprocess Adjacency Matrix for use with layer_graph_conv
#'
#'This utility function can be used to  preprocess a network adjacency matrix
#'into an object that can be used to represent the network in the \code{\link{layer_graph_conv}} layer.
#'Internally it does this:\cr
#'	\code{\link{utils_localpooling_filter}(A)}\cr
#'
#'@param A An Adjacency matrix (can be dense or sparse)
#'
#'@export
preprocess_graph_conv <- function(A) {
	spk$layers$GraphConv$preprocess(A)
}

#' Preprocess Adjacency Matrix for use with layer_graph_conv_skip
#'
#'This utility function can be used to  preprocess a network adjacency matrix
#'into an object that can be used to represent the network in the \code{\link{layer_graph_conv_skip}} layer.
#'Internally it does this:\cr
#'	\code{\link{utils_normalized_adjacency}(A)}\cr
#'
#'@param A An Adjacency matrix (can be dense or sparse)
#'
#'@export
preprocess_graph_conv_skip <- function(A) {
	spk$layers$GraphConvSkip$preprocess(A)
}

#' Preprocess Adjacency Matrix for use with layer_tag_conv
#'
#'This utility function can be used to  preprocess a network adjacency matrix
#'into an object that can be used to represent the network in the \code{\link{layer_tag_conv}} layer.
#'Internally it does this:\cr
#'	\code{\link{utils_normalized_adjacency}(A)}\cr
#'
#'@param A An Adjacency matrix (can be dense or sparse)
#'
#'@export
preprocess_tag_conv <- function(A) {
	spk$layers$TAGConv$preprocess(A)
}

