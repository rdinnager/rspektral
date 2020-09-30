## code to prepare `build_package` dataset goes here

library(dplyr)
library(readr)
library(purrr)
library(stringr)
library(unglue)

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
               head(-1) %>%
               tail(-1)) %>%
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

## fix up docos
conv_titles <- conv_docs[ , 2] %>%
  stringr::str_extract("(.*?)layer")
conv_docs_fixed <- conv_docs[ , 2] %>%



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
                                regex('r\\"\\"\\"(.*?)\\"\\"\\"',
                                      dotall = TRUE))

pool_params_code <- stringr::str_match(pool_layer_source,
                                  regex('def __init__\\((.*?)\\)',
                                        dotall = TRUE)) %>%
  .[ , 2] %>%
  stringr::str_remove_all("\n") %>%
  stringr::str_split(",") %>%
  purrr::map(~stringr::str_trim(.x) %>%
               head(-1) %>%
               tail(-1)) %>%
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

layer_param_dat <- layer_param_dat %>%
  dplyr::left_join(tibble::tribble(~var, ~is_integer,
                                   "channels", TRUE,
                                   "k", TRUE)) %>%
  dplyr::mutate(is_integer = ifelse(is.na(is_integer), FALSE, is_integer)) %>%
  dplyr::mutate(default = dplyr::case_when(default == "True" ~ "TRUE",
                                           default == "False" ~ "FALSE",
                                           default == "None" ~ "NULL",
                                           TRUE ~ default)) %>%
  dplyr::mutate(is_numeric = !is.na(as.numeric(default)))

usethis::use_data(build_package, overwrite = TRUE)
