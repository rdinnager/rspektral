#' Citation datasets
#'
#' Loads a citation dataset (Cora, Citeseer or Pubmed) using the "Planetoid"
#' splits initially defined in [Yang et al. (2016)](https://arxiv.org/abs/1603.08861).
#' The train, test, and validation splits are given as binary masks. Node attributes are bag-of-words vectors representing the most common words
#' in the text document associated to each node.
#' Two papers are connected if either one cites the other.
#' Labels represent the class of the paper.
#' @param dataset_name name of the dataset to load (`'cora'`, `'citeseer'`, or
#' `'pubmed'`);
#' @param normalize_features normalize_features normalize_features: if TRUE,
#' the node features are normalized;
#' @param random_split random_split if TRUE, return a randomized split (20 nodes per class
#' for training, 30 nodes per class for validation and the remaining nodes for
#' testing, [Shchur et al. (2018)](https://arxiv.org/abs/1811.05868))
#' @return_type Data format to return data in. One of either "list", or "tidygraph"
#'
#' @return Either a list with 6 elements (containing - Adjacency matrix, Node features,
#' Labels, and 3 binary masks for train, validation, and test splits), or a \code{tbl_graph}
#' object with the Node features, Labels, and 3 binary masks as node attributes).
#' @export
dataset_citations <- function(dataset_name = "cora", normalize_features = TRUE, random_split = FALSE,
                              return_type = c("list", "tidygraph")) {

  return_type <- match.arg(return_type)

  args <- list(dataset_name = dataset_name,
               normalize_features = normalize_features,
               random_split = random_split)

  dat <- do.call(spk$datasets$citation$load_data, args)

  if(return_type == "tidygraph") {
    stop("Sorry tidygraph return type has not been implemented yet")
  }

  dat

}

#' Generate Delaunay triangulation network data
#'
#' Generates a dataset of Delaunay triangulations as described by [Zambon et al. (2017)](https://arxiv.org/abs/1706.06941).
#' Node attributes are the 2D coordinates of the points.
#' Two nodes are connected if they share an edge in the Delaunay triangulation.
#' Labels represent the class of the graph (0 to 20, each class index i
#' represent the "difficulty" of the classification problem 0 v. i. In other
#' words, the higher the class index, the more similar the class is to class 0).
#' @return
#' - if `return_type='list'`, the adjacency matrix, node features, and
#' an array containing labels, in a list
#' - if `return_type='tidygraph'`, a tidygraph object, with node features and labels as
#' node data
#'
#' @param classes classes indices of the classes to load (integer, or list of integers
#' between 0 and 20)
#' @param n_samples_in_class number of generated samples per class
#' @param n_nodes n_nodes number of nodes in a graph
#' @param support_low support_low lower bound of the uniform distribution from which the
#' support is generated
#' @param support_high support_high upper bound of the uniform distribution from which the
#' support is generated
#' @param drift_amount drift_amount coefficient to control the amount of change between classes
#' @param one_hot_labels one_hot_labels one-hot encode dataset labels
#' @param support support custom support to use instead of generating it randomly
#' @param seed seed random numpy seed
#' @param return_type Data format to return data in. One of either "list", or "tidygraph"
#'
#' @export
dataset_delaunay_generate <- function(classes = 0L, n_samples_in_class = 1000L, n_nodes = 7L, support_low = 0.0, support_high = 10.0, drift_amount = 1.0, one_hot_labels = TRUE, support = NULL, seed = NULL, return_type = "numpy") {

  dat <- spk$datasets$delaunay$generate_data(
    classes = as.integer(classes),
    n_samples_in_class = as.integer(n_samples_in_class),
    n_nodes = as.integer(n_nodes),
    support_low = support_low,
    support_high = support_high,
    drift_amount = drift_amount,
    one_hot_labels = one_hot_labels,
    support = support,
    seed = as.integer(seed),
    return_type = "numpy"
  )

  if(return_type == "tidygraph") {
    stop("Sorry tidygraph return type has not been implemented yet")
  }

  dat
}

#' GraphSage Datasets
#'
#' Loads one of the datasets (PPI or Reddit) used in [Hamilton & Ying (2017)](https://arxiv.org/abs/1706.02216).
#' The PPI dataset (originally [Stark et al. (2006)](https://www.ncbi.nlm.nih.gov/pubmed/16381927))
#' for inductive node classification uses positional gene sets, motif gene sets
#' and immunological signatures as features and gene ontology sets as labels.
#' The Reddit dataset consists of a graph made of Reddit posts in the month of
#' September, 2014. The label for each node is the community that a
#' post belongs to. The graph is built by sampling 50 large communities and
#' two nodes are connected if the same user commented on both. Node features
#' are obtained by concatenating the average GloVe CommonCrawl vectors of
#' the title and comments, the post's score and the number of comments.
#' The train, test, and validation splits are returned as binary masks.
#' :param max_degree: int, if positive, subsample edges so that each node has
#' the specified maximum degree.
#' :param normalize_features: if TRUE, the node features are normalized;
#' :
#'
#' @param dataset_name dataset_name name of the dataset to load (`'ppi'`, or `'reddit'`)
#' @param max_degree max_degree if positive, subsample edges so that each node has
#' the specified maximum degree.
#' @param normalize_features normalize_features if TRUE, the node features are normalized
#' @param return_type Data format to return data in. One of either "list", or "tidygraph"
#'
#' @return Either a list with 6 elements (containing - Adjacency matrix, Node features,
#' Labels, and 3 binary masks for train, validation, and test splits), or a \code{tbl_graph}
#' object with the Node features, Labels, and 3 binary masks as node attributes).
#'
#' @export
dataset_graphsage <- function(dataset_name, max_degree = -1L, normalize_features = TRUE) {

  dat <- spk$datasets$graphsage$load_data(
    dataset_name = dataset_name,
    max_degree = max_degree,
    normalize_features = normalize_features
  )

  if(return_type == "tidygraph") {
    stop("Sorry tidygraph return type has not been implemented yet")
  }

  dat

}

#' MNIST Graph Dataset
#'
#' Loads the MNIST dataset and a K-NN graph to perform graph signal classification, as described by [Defferrard et al. (2016)](https://arxiv.org/abs/1606.09375).
#' The K-NN graph is statically determined from a regular grid of pixels using
#' the 2d coordinates. The node features of each graph are the MNIST digits vectorized and rescaled
#' to [0, 1].
#' Two nodes are connected if they are neighbours according to the K-NN graph.
#' Labels are the MNIST class associated to each sample.
#' @param k int, number of neighbours for each node
#' @param noise_level fraction of edges to flip (from 0 to 1 and vice versa)
#' @param return_type Data format to return data in. One of either "list", or "tidygraph"
#'
#' @return - X_train, y_train: training node features and labels; - X_val, y_val: validation node features and labels; - X_test, y_test: test node features and labels; - A: adjacency matrix of the grid;
#'
#' @export
dataset_graph_mnist <- function(k = 8L, noise_level = 0.0) {
  dat <- spk$datasets$mnist$load_data(
    k = as.integer(k),
    noise_level = noise_level
  )

  if(return_type == "tidygraph") {
    stop("Sorry tidygraph return type has not been implemented yet")
  }

  dat

}


#' QM9 Small Molecule Chemical Dataset
#'
#' Loads the QM9 chemical data set of small molecules.
#' Nodes represent heavy atoms (hydrogens are discarded), edges represent
#' chemical bonds. The node features represent the chemical properties of each atom, and are
#' loaded according to the `nf_keys` argument.
#' See `rspektral_datasets$qm9$NODE_FEATURES` for possible node features, and
#' see [this link](http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx)
#' for the meaning of each property. Usually, it is sufficient to load the
#' atomic number. The edge features represent the type and stereoscopy of each chemical bond
#' between two atoms.
#' See `rspektral_datasets$qm9$EDGE_FEATURES` for possible edge features, and
#' see [this link](http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx)
#' for the meaning of each property. Usually, it is sufficient to load the
#' type of bond.
#' @param nf_keys list or str, node features to return (see `rspektral_datasets$qm9$NODE_FEATURES` for available features);
#' @param ef_keys list or str, edge features to return (see `rspektral_datasets$qm9$EDGE_FEATURES` for available features);
#' @param auto_pad if `return_type='numpy'`, zero pad graph matrices to have the same number of nodes
#' @param self_loops: add self loops to adjacency matrices
#' @param amount the amount of molecules to return (in ascending order by
#' number of atoms).
#' @param return_type Data format to return data in. One of either "list", or "tidygraph"
#' @return
#' - if `return_type='list'`, a list of the adjacency matrix, node features, edge features,
#' and a dataframe containing labels
#' - if `return_type="tidygraph`, a list tidygraph objects with node and edge features as
#' node and edge data columns, labels will be the names of the lists's elements
#'
#' @export
dataset_qm9 <- function(nf_keys = NULL, ef_keys = NULL, auto_pad = TRUE, self_loops = FALSE, amount = NULL, return_type = "numpy") {
  dat <- spk$datasets$qm9$load_data(
    nf_keys = nf_keys,
    ef_keys = ef_keys,
    auto_pad = auto_pad,
    self_loops = self_loops,
    amount = amount,
    return_type = return_type
  )

  if(return_type == "tidygraph") {
    stop("Sorry tidygraph return type has not been implemented yet")
  }

  dat

}


#' TU Dortmund Benchmark Dataset
#'
#' Loads one of the Benchmark Data Sets for Graph Kernels from TU Dortmund
#' ([link](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).
#' The node features are computed by concatenating the following features for
#' each node: - node attributes, if available, normalized as specified in `normalize_features`;
#' - clustering coefficient, normalized with z-score;
#' - node degrees, normalized as specified in `normalize_features`;
#' - node labels, if available, one-hot encoded.
#' @param dataset_name name of the dataset to load (see `spektral_datasets$tud$AVAILABLE_DATASETS`).
#' @param clean if TRUE, return a version of the dataset with no isomorphic graphs.
#' @return A list of:
#' - a list of adjacency matrices;
#' - a list of node feature matrices;
#' - an array containing the one-hot encoded targets.
#'
#' @export
dataset_tud <- function(dataset_name, clean = FALSE) {
  dat <- spk$datasets$tud$load_data(
    dataset_name = dataset_name,
    clean = clean
  )

  dat

}
