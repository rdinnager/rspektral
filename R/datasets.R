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
    dat <- convert_to_tidygraph(dat)
  }

  dat

}

convert_to_tidygraph <- function(dat) {
  tg <- tidygraph::as_tbl_graph(dat[[1]]) %>%
    tidygraph::activate(nodes) %>%
    tidygraph::mutate(features = dat[[2]],
                      labels = dat[[3]],
                      train_mask = dat[[4]],
                      validation_mask = dat[[5]],
                      test_mask = dat[[6]])
}
