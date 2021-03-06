% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/preprocess.R
\name{preprocess_cheb_conv}
\alias{preprocess_cheb_conv}
\title{Preprocess Adjacency Matrix for use with layer_cheb_conv}
\usage{
preprocess_cheb_conv(A)
}
\arguments{
\item{A}{An Adjacency matrix (can be dense or sparse)}
}
\description{
This utility function can be used to  preprocess a network adjacency matrix
into an object that can be used to represent the network in the \code{\link{layer_cheb_conv}} layer.
Internally it does this:\cr
\code{L=\link{utils_normalized_laplacian}(A)}\cr
\code{L=\link{utils_rescale_laplacian}(L)}\cr
\code{L}\cr
}
