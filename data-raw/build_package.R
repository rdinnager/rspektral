## code to prepare `build_package` dataset goes here

library(dplyr)
library(readr)
library(purrr)
library(stringr)
library(unglue)
library(snakecase)
library(reticulate)
library(tidygraph)
library(dplyr)

spektral_dir <- tempdir()
src_dir <- usethis::use_course("danielegrattarola/spektral", spektral_dir)

1
1

conv_layer_source <- file.path(src_dir, "spektral/layers/convolutional") %>%
  list.files(full.names = TRUE) %>%
  .[-1] %>%
  purrr::map_chr(~readr::read_lines(.x) %>%
                   paste(collapse = "\n")) %>%
  stringr::str_split("\n\n\n") %>%
  purrr::map(~.x[-1]) %>%
  unlist()

conv_layer_names <- stringr::str_match(conv_layer_source,
                                       'class[:blank:](.*?)\\((.*?)\\)')

conv_docs <- stringr::str_match(conv_layer_source,
                                    regex('r\\"\\"\\"(.*?)\\"\\"\\"',
                                          dotall = TRUE))

params_code <- stringr::str_match(conv_layer_source,
                                      regex('def __init__\\((.*?)\\)',
                                            dotall = TRUE)) %>%
  .[ , 2] %>%
  .[-16] %>%
  stringr::str_remove_all("\n") %>%
  stringr::str_split(",") %>%
  purrr::map(~stringr::str_trim(.x) %>%
               {.[. != "self"]} %>%
               {.[!grepl("kwargs", .)]}) %>%
  setNames(conv_layer_names[ , 2][-16]) %>%
  purrr::imap_dfr(~unglue::unglue_data(.x,
                                       c("{var}={default}",
                                         "{var}")) %>%
                    dplyr::mutate(layer_name = .y,
                                  var = stringr::str_remove_all(var,
                                                                ": int") %>%
                                    stringr::str_trim()))

params_doc <- conv_docs[ , 2] %>%
  .[-16] %>%
  stringr::str_match(regex("\\*\\*Arguments\\*\\*(.*?)-(.*)$",
                           dotall = TRUE)) %>%
  .[ , 3] %>%
  stringr::str_split("- ") %>%
  purrr::map(~stringr::str_split(.x,
                                      ": ")) %>%
  purrr::map(~purrr::map_chr(.x, 2) %>%
               stringr::str_remove_all(";") %>%
               stringr::str_trim() %>%
               setNames(purrr::map_chr(.x, 1) %>%
                          stringr::str_remove_all("`") %>%
                          stringr::str_trim())) %>%
  setNames(conv_layer_names[ , 2][-16]) %>%
  purrr::imap_dfr(~dplyr::tibble(var = names(.x),
                                 description = .x) %>%
                    dplyr::mutate(layer_name = .y))



layer_param_dat <- params_code %>%
  dplyr::left_join(params_doc)

unique(layer_param_dat$var)

layer_param_dat <- layer_param_dat %>%
  dplyr::left_join(tibble::tribble(~var, ~is_integer,
                                   "channels", TRUE,
                                   "propagations", TRUE,
                                   "mlp_hidden", TRUE,
                                   "order", TRUE,
                                   "iterations", TRUE,
                                   "K", TRUE,
                                   "num_diffusion", TRUE,
                                   "n_layers", TRUE,
                                   "attn_heads", TRUE,
                                   "concat_heads", TRUE)) %>%
  dplyr::mutate(is_integer = ifelse(is.na(is_integer), FALSE, is_integer)) %>%
  dplyr::mutate(default = dplyr::case_when(default == "True" ~ "TRUE",
                                           default == "False" ~ "FALSE",
                                           default == "None" ~ "NULL",
                                           TRUE ~ default)) %>%
  dplyr::mutate(is_numeric = !is.na(as.numeric(default)))

## fix up docos (mainly to get equations to work with mathjaxr)
conv_titles <- conv_layer_names[ , 2]

conv_docs_fixed <- conv_docs[ , 2] %>%
  .[-16] %>%
  stringr::str_replace_all(regex("\\\\\\((.*?)\\\\\\)",
                                 dotall = TRUE),
                           "\\\\mjeqn{\\1}{}") %>%
  stringr::str_replace_all(regex("\\$\\$(.*?)\\$\\$",
                                 dotall = TRUE),
                           function(x) x %>%
                             stringr::str_replace_all("\n[:blank:]*", " ")) %>%
  stringr::str_replace_all(regex("\\$\\$(.*?)\\$\\$",
                               dotall = TRUE),
                           "\\\\mjdeqn{\\1}{}") %>%
  # stringr::str_replace_all(regex("\\\\([:alpha:]{1})([^[:alnum:]])",
  #                                dotall = TRUE,
  #                                multiline = TRUE),
  #                          "\\\\boldsymbol{\\1}\\2") %>%
  # stringr::str_replace_all(regex("\\\\([:upper:]{1})",
  #                                dotall = TRUE),
  #                          "\\\\boldsymbol{\\1}") %>%
  stringr::str_replace_all(regex("\\\\([:alpha:]{1})([^[:alnum:]])",
                                 dotall = TRUE,
                                 multiline = TRUE),
                           "\\1\\2") %>%
  stringr::str_replace_all(regex("\\\\([:upper:]{1})",
                                 dotall = TRUE),
                           "\\1") %>%
  stringr::str_remove(regex("\\*\\*Arguments\\*\\*(.*?)-(.*)$",
                           dotall = TRUE)) %>%
  ## edge cases
  #stringr::str_replace_all("\\\\Z", "Z") %>%
  stringr::str_replace_all("\\\\bar([^[:blank:]])", "\\\\bar \\1") %>%
  stringr::str_replace_all("\\\\hat([^[:blank:]])", "\\\\hat \\1") %>%
  stringr::str_replace_all("(\\S+)\\\\mjeqn\\{_(.*?)\\}",
                           "\\\\mjeqn{\\\\mathrm{\\1} _ \\2}") %>%
  stringr::str_replace_all("\\}_", "} _ ") %>%
  stringr::str_replace_all("\\\\big\\[", "[") %>%
  stringr::str_replace_all("\\\\big\\]", "]") %>%
  ## add roxygen tags
  stringr::str_replace_all("\n[:blank:]*",
                           "\n#' ") %>%
  paste0("#' @description \\loadmathjax ", .) %>%
  paste0("#' ", conv_titles[-16], "\n#' \n", .) %>%
  dplyr::tibble(layer_name = conv_titles[-16],
                roxy = .)

conv_param_roxy <- layer_param_dat %>%
  dplyr::group_by(layer_name) %>%
  dplyr::summarise(param_roxy = paste0("@param ", var, " ", description) %>%
                     paste(collapse = "\n") %>%
                     paste("#'", .) %>%
                     stringr::str_replace_all("\n[:blank:]*",
                                              "\n#' "),
                   .groups = "drop")

conv_docs_fixed <- conv_docs_fixed %>%
  dplyr::left_join(conv_param_roxy) %>%
  dplyr::mutate(roxy = paste(roxy, param_roxy,
                             "#' @export",
                             sep = "\n"))


##### pooling ######

pool_layer_source <- file.path(src_dir, "spektral/layers/pooling") %>%
  list.files(full.names = TRUE) %>%
  .[-1] %>%
  purrr::map_chr(~readr::read_lines(.x) %>%
                   paste(collapse = "\n")) %>%
  stringr::str_split("\n\n\n") %>%
  purrr::map(~.x[-1]) %>%
  unlist()

pool_layer_names <- stringr::str_match(pool_layer_source,
                                       'class[:blank:](.*?)\\((.*?)\\)')

pool_docs <- stringr::str_match(pool_layer_source,
                                regex('\\"\\"\\"(.*?)\\"\\"\\"',
                                      dotall = TRUE))

pool_params_code <- stringr::str_match(pool_layer_source,
                                  regex('def __init__\\((.*?)\\)',
                                        dotall = TRUE)) %>%
  .[ , 2] %>%
  stringr::str_remove_all("\n") %>%
  stringr::str_split(",") %>%
  purrr::map(~stringr::str_trim(.x) %>%
               {.[. != "self"]} %>%
               {.[!grepl("kwargs", .)]}
             ) %>%
  purrr::map(~{if(length(.x) == 0) "None"
                else .x}) %>%
  setNames(pool_layer_names[ , 2]) %>%
  purrr::imap_dfr(~unglue::unglue_data(.x,
                                       c("{var}={default}",
                                         "{var}")) %>%
                    dplyr::mutate(layer_name = .y))

pool_params_doc <- pool_docs[ , 2] %>%
  stringr::str_match(regex("\\*\\*Arguments\\*\\*(.*?)-(.*)$",
                           dotall = TRUE)) %>%
  .[ , 3] %>%
  stringr::str_split("- ") %>%
  purrr::map(~stringr::str_split(.x,
                                 ": ")) %>%
  purrr::map(~{if(is.na(.x[[1]][1])) list(c("None", "No Arguments"))
                  else .x}) %>%
  purrr::map(~purrr::map_chr(.x, 2) %>%
               stringr::str_remove_all(";") %>%
               stringr::str_trim() %>%
               setNames(purrr::map_chr(.x, 1) %>%
                          stringr::str_remove_all("`") %>%
                          stringr::str_trim())) %>%
  setNames(pool_layer_names[ , 2]) %>%
  purrr::imap_dfr(~dplyr::tibble(var = names(.x),
                                 description = .x) %>%
                    dplyr::mutate(layer_name = .y))

pool_param_dat <- pool_params_code %>%
  dplyr::left_join(pool_params_doc)

unique(pool_param_dat$var)

pool_param_dat <- pool_param_dat %>%
  dplyr::left_join(tibble::tribble(~var, ~is_integer,
                                   "channels", TRUE,
                                   "k", TRUE)) %>%
  dplyr::mutate(is_integer = ifelse(is.na(is_integer), FALSE, is_integer)) %>%
  dplyr::mutate(default = dplyr::case_when(default == "True" ~ "TRUE",
                                           default == "False" ~ "FALSE",
                                           default == "None" ~ "NULL",
                                           TRUE ~ default)) %>%
  dplyr::mutate(is_numeric = !is.na(as.numeric(default)))


## fix up docos (mainly to get equations to work with mathjaxr)
pool_titles <- pool_layer_names[ , 2]

pool_docs_fixed <- pool_docs[ , 2] %>%
  stringr::str_replace_all(regex("\\\\\\((.*?)\\\\\\)",
                                 dotall = TRUE),
                           "\\\\mjeqn{\\1}{}") %>%
  stringr::str_replace_all(regex("\\$\\$(.*?)\\$\\$",
                                 dotall = TRUE),
                           function(x) x %>%
                             stringr::str_remove_all("\n[:blank:]*")) %>%
  stringr::str_replace_all(regex("\\$\\$(.*?)\\$\\$",
                                 dotall = TRUE),
                           "\\\\mjdeqn{\\1}{}") %>%
  stringr::str_replace_all(regex("\\\\([:alpha:]{1})([^[:alnum:]])",
                                 dotall = TRUE,
                                 multiline = TRUE),
                           "\\\\boldsymbol{\\1}\\2") %>%
  stringr::str_remove(regex("\\*\\*Arguments\\*\\*(.*?)-(.*)$",
                            dotall = TRUE)) %>%
  ## edge cases
  #stringr::str_replace_all("\\\\Z", "Z") %>%
  stringr::str_replace_all("\\\\bar([^[:blank:]])", "\\\\bar \\1") %>%
  stringr::str_replace_all("\\\\hat([^[:blank:]])", "\\\\hat \\1") %>%
  stringr::str_replace_all("(\\S+)\\\\mjeqn\\{_(.*?)\\}",
                           "\\\\mjeqn{\\\\mathrm{\\1} _ \\2}") %>%
  stringr::str_replace_all("\\}_", "} _ ") %>%
  stringr::str_replace_all("\\|_", "\\| _ ") %>%
  stringr::str_replace_all("\\\\big\\[", "[") %>%
  stringr::str_replace_all("\\\\big\\]", "]") %>%
  ## add roxygen tags
  stringr::str_replace_all("\n[:blank:]*",
                           "\n#' ") %>%
  paste0("#' @description \\loadmathjax ", .) %>%
  paste0("#' ", pool_titles, "\n#' \n", .) %>%
  dplyr::tibble(layer_name = pool_titles,
                roxy = .)

pool_param_roxy <- pool_param_dat %>%
  dplyr::group_by(layer_name) %>%
  dplyr::summarise(param_roxy = paste0("@param ", var, " ", description) %>%
                     paste(collapse = "\n") %>%
                     paste("#'", .) %>%
                     stringr::str_replace_all("\n[:blank:]*",
                                              "\n#' "),
                   .groups = "drop") %>%
  dplyr::mutate(param_roxy = ifelse(grepl("None", param_roxy),
                                    "", param_roxy))

pool_docs_fixed <- pool_docs_fixed %>%
  dplyr::left_join(pool_param_roxy) %>%
  dplyr::mutate(roxy = paste(roxy, param_roxy,
                             "#' @export",
                             sep = "\n"))


########## preprocess functions ##########

preproc_source <- conv_layer_source %>%
  .[-16] %>%
  stringr::str_match(regex("def preprocess\\((.*?)\\)\\:(.*?)$",
                           dotall = TRUE))

non_triv <- stringr::str_which(preproc_source[ , 3],
                               "\n        return A",
                               negate = TRUE)

########### datasets #########



########## start generating R files ###########

conv_layer_code <- layer_param_dat %>%
  dplyr::mutate(func_name = paste0("layer_", snakecase::to_snake_case(layer_name))) %>%
  dplyr::group_by(layer_name) %>%
  dplyr::summarise(header = paste0(func_name[1], " <- function(object,\n\t",
                          paste0(var, ifelse(is.na(default), "", paste0(" = ", default)),
                                collapse = ",\n\t"),
                          ",\n\t...)\n"),
                   body = paste0("{\n\targs <- list(",
                                 paste0(var, " = ",
                                        ifelse(!is_integer, var, paste0("as.integer(", var, ")")),
                                        collapse = ",\n\t\t"),
                                 "\n\t\t)\n\tkeras::create_layer(spk$layers$",
                                 layer_name[1],
                                 ", object, args)\n}\n"),
                   func_name = func_name[1],
                   .groups = "drop") %>%
  dplyr::mutate(code = paste0(header, body)) %>%
  dplyr::left_join(conv_docs_fixed) %>%
  dplyr::mutate(all_text = paste(roxy, code, sep = "\n"))

pool_layer_code <- pool_param_dat %>%
  dplyr::mutate(var = ifelse(var == "None", "", var)) %>%
  dplyr::mutate(func_name = paste0("layer_", snakecase::to_snake_case(layer_name))) %>%
  dplyr::group_by(layer_name) %>%
  dplyr::summarise(header = paste0(func_name[1], " <- function(object,\n\t",
                                   paste0(var, ifelse(is.na(default), "", paste0(" = ", default)),
                                          collapse = ",\n\t"),
                                   ifelse(var[1] == "", "", ",\n"),
                                   "\t...)\n"),
                   body = paste0("{\n\targs <- list(",
                                 paste0(var, ifelse(var == "", "", " = "),
                                        ifelse(!is_integer, var, paste0("as.integer(", var, ")")),
                                        collapse = ",\n\t\t"),
                                 "\n\t\t)\n\tkeras::create_layer(spk$layers$",
                                 layer_name[1],
                                 ", object, args)\n}\n"),
                   func_name = func_name[1],
                   .groups = "drop") %>%
  dplyr::mutate(code = paste0(header, body)) %>%
  dplyr::left_join(pool_docs_fixed) %>%
  dplyr::mutate(all_text = paste(roxy, code, sep = "\n"))



preproc_code <- paste0(conv_layer_code$func_name[non_triv] %>%
                         stringr::str_replace("layer_", "preprocess_"),
                       " <- function(A) {\n\tspk$layers$", conv_layer_code$layer_name[non_triv],
                       "$preprocess(A)\n}\n")

preproc_docs <- "#' Preprocess Adjacency Matrix for use with <layer_name>
#'
#'This utility function can be used to  preprocess a network adjacency matrix
#'into an object that can be used to represent the network in the \\code{\\link{<layer_name>}} layer.
#'Internally it does this:\\cr<code>
#'
#'@param A An Adjacency matrix (can be dense or sparse)
#'
#'@export
"

preproc_docs <- glue::glue_data(list(layer_name = conv_layer_code$func_name[non_triv],
                                     code = preproc_source[non_triv, 3] %>%
                                       stringr::str_remove_all("return ") %>%
                                       stringr::str_remove_all("[:blank:]") %>%
                                       stringr::str_replace_all("(\\w+)\\((.*?)\\)",
                                                              "\\\\link{utils_\\1}(\\2)") %>%
                                       stringr::str_replace_all("\n", "\n#'\t\\\\code{") %>%
                                       stringr::str_replace_all(regex("$", multiline = TRUE),
                                                                "}\\\\cr") %>%
                                       stringr::str_remove_all("^\\}\\\\cr")),
                                preproc_docs,
                                .open = "<",
                                .close = ">")

preproc <- paste(preproc_docs, preproc_code, sep = "\n")

readr::write_lines(conv_layer_code$all_text,
                  "R/layers_conv.R")

readr::write_lines(pool_layer_code$all_text,
                   "R/layers_pool.R")

readr::write_lines(preproc,
                   "R/preprocess.R")
