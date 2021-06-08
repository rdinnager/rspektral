#' Compute normalized Laplacian of an adjacency matrix
#'
#' @description \loadmathjax
#' Computes a normalized Laplacian of the given adjacency matrix as
#' \mjeqn{I - D^{-1}A}{I - D^(-1)A} or \mjeqn{I - D^{-1/2}AD^{-1/2}}{I - D^(-1/2)AD^(-1/2)} (symmetric normalization).
#' @param A rank 2 array or sparse matrix;
#' @param symmetric boolean, compute symmetric normalization
#' @return the normalized Laplacian.
#'
#' @export
utils_normalized_laplacian <- function(A, symmetric = TRUE) {
  spk$utils$normalized_laplacian(
    A = A,
    symmetric = symmetric
  )
}

#' Rescale a Laplacian
#'
#' @description \loadmathjax
#' Rescales the Laplacian eigenvalues in [-1,1], using lmax as largest eigenvalue.
#'
#' @param L  rank 2 array or sparse matrix
#' @param lmax if NULL, compute largest eigenvalue with scipy.linalg.eisgh.
#' If the eigendecomposition fails, lmax is set to 2 automatically.
#' If scalar, use this value as largest eignevalue when rescaling.
#' @return the rescaled Laplacian
#'
#' @export
utils_rescale_laplacian <- function(L, lmax = NULL) {
  spk$utils$rescale_laplacian(
    L = L,
    lmax = lmax
  )
}

#' Compute Graph Filter
#'
#' @description \loadmathjax
#' Computes the graph filter described in [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).
#' @param A array or sparse matrix with rank 2 or 3
#' @param symmetric boolean, whether to normalize the matrix as \mjeqn{D^{-\frac{1}{2}}AD^{-\frac{1}{2}}}{D^(-1/2)AD^(-1/2)} or as \mjeqn{D^{-1}A}{D^(-1)A}
#' @return array or sparse matrix with rank 2 or 3, same as A
#'
#' @export
utils_localpooling_filter <- function(A, symmetric = TRUE) {
  spk$utils$localpooling_filter(
    A = A,
    symmetric = symmetric
  )
}


#' Normalize Adjacency Matrix
#'
#' @description \loadmathjax
#' Normalizes the given adjacency matrix using the degree matrix as either
#' \mjeqn{D^{-1}A}{D^(-1)A} or \mjeqn{D^{-1/2}AD^{-1/2}}{D^(-1/2)AD^(-1/2)} (symmetric normalization).
#' @param A rank 2 array or sparse matrix
#' @param symmetric boolean, compute symmetric normalization
#' @return the normalized adjacency matrix.
#'
#' @export
utils_normalized_adjacency <- function(A, symmetric = TRUE) {
  spk$utils$normalized_adjacency(
    A = A,
    symmetric = symmetric
  )
}
