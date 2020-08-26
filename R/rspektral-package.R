#' @keywords internal
"_PACKAGE"

# The following block is used by usethis to automatically manage
# roxygen namespace tags. Modify with care!
## usethis namespace: start
## usethis namespace: end
NULL

spk <- NULL

.onLoad <- function(libname, pkgname) {

  spk <<- reticulate::import("spektral", delay_load = list(
    priority = 10,
    environment = "r-tensorflow"
  ))

}
