#' Calculate modified Laplacian on an adjacency matrix
#'
#' This is used to preprocess and adjacency matrix so it can be used directly
#' in a graph convolutional network (GCN), as implement by \code{\link{layer_graph_conv}}
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


#' Calculate Chebyshev polynomials from an adjacency matrix
#'
#' This is used to preprocess and adjacency matrix so it can be used directly
#' in a Chebyshev graph convolutional layer, as implement by \code{\link{layer_cheb_conv}}
#'
#' @param A Adjacency matrix as a sparse matrix (\code{\link[Matrix]{dgRMatrix}})
#' @param densify Should the result be returned as a dense matrix (default is sparse)?
#'
#' @return Chebyshev polynomials as a sparse matrix (or a dense matrix if \code{densify} is \code{TRUE})
#' @export
preprocess_chebyshev <- function(A, densify = FALSE) {
  res <- spk$layers$ChebConv$preprocess(A)
  if(densify) {
    res <- as.matrix(res)
  }

  res

}
