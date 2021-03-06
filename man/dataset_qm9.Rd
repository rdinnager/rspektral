% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/datasets.R
\name{dataset_qm9}
\alias{dataset_qm9}
\title{QM9 Small Molecule Chemical Dataset}
\usage{
dataset_qm9(
  nf_keys = NULL,
  ef_keys = NULL,
  auto_pad = TRUE,
  self_loops = FALSE,
  amount = NULL,
  return_type = "numpy"
)
}
\arguments{
\item{nf_keys}{list or str, node features to return (see \code{rspektral_datasets$qm9$NODE_FEATURES} for available features);}

\item{ef_keys}{list or str, edge features to return (see \code{rspektral_datasets$qm9$EDGE_FEATURES} for available features);}

\item{auto_pad}{if \code{return_type='numpy'}, zero pad graph matrices to have the same number of nodes}

\item{amount}{the amount of molecules to return (in ascending order by
number of atoms).}

\item{return_type}{Data format to return data in. One of either "list", or "tidygraph"}

\item{self_loops:}{add self loops to adjacency matrices}
}
\value{
\itemize{
\item if \code{return_type='list'}, a list of the adjacency matrix, node features, edge features,
and a dataframe containing labels
\item if \verb{return_type="tidygraph}, a list tidygraph objects with node and edge features as
node and edge data columns, labels will be the names of the lists's elements
}
}
\description{
Loads the QM9 chemical data set of small molecules.
Nodes represent heavy atoms (hydrogens are discarded), edges represent
chemical bonds. The node features represent the chemical properties of each atom, and are
loaded according to the \code{nf_keys} argument.
See \code{rspektral_datasets$qm9$NODE_FEATURES} for possible node features, and
see \href{http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx}{this link}
for the meaning of each property. Usually, it is sufficient to load the
atomic number. The edge features represent the type and stereoscopy of each chemical bond
between two atoms.
See \code{rspektral_datasets$qm9$EDGE_FEATURES} for possible edge features, and
see \href{http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx}{this link}
for the meaning of each property. Usually, it is sufficient to load the
type of bond.
}
