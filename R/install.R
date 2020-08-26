#' Install spektral keras extension library in Python
#'
#' @param version WHat version of spektral to install? Default is to install latest version
#' available in pip
#' @param ... Further arguments passed to \code{\link[reticulate]{py_install}}
#' @param restart_session Should the session be restarted after installation?
#' @export
install_spektral <- function(version = NULL, ..., restart_session = TRUE) {

  if (is.null(version))
    module_string <- "spektral"
  else
    module_string <- paste0("spektral==", version)

  invisible(reticulate::py_config())
  reticulate::py_install(packages = paste(module_string), pip = TRUE, ...)

  if(requireNamespace("rstudioapi", quietly = TRUE)) {
    if (restart_session && rstudioapi::hasFun("restartSession"))
      rstudioapi::restartSession()
  }
}
