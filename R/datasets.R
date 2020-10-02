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
#'
#' @return Either a list with 6 elements (containing - Adjacency matrix, Node features,
#' Labels, and 3 binary masks for train, validation, and test splits), or a \code{tbl_graph}
#' object with the Node features, Labels, and 3 binary masks as node attributes).
#'
#' @export
load_data <- function(dataset_name, max_degree = -1L, normalize_features = TRUE) {

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
