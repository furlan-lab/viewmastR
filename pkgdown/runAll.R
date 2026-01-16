# quick testing -----------------------------------------------------------
pkgdown::clean_site(pkg = ".")
pkgdown::init_site(pkg = ".")
pkgdown::build_home_index()
pkgdown::preview_page("index.html")
pkgdown::build_article(name = "Extras")
pkgdown::preview_page("articles/Extras.html")

# cleanup start -----------------------------------------------------------
pkgdown::clean_site(pkg = ".")
pkgdown::init_site(pkg = ".")

# index -------------------------------------------------------------------
pkgdown::build_home(preview = TRUE)
pkgdown::build_news(preview = TRUE)

# reference ---------------------------------------------------------------
# source("pkgdown/02-pkgdown-add-to-yalm-reference.R")
pkgdown::build_reference_index()
pkgdown::build_reference()
pkgdown::preview_site(path = "/reference")


# rticles -----------------------------------------------------------------
options(rmarkdown.html_vignette.check_title = FALSE)
pkgdown::build_article("HowTo")
pkgdown::build_article("InDepth")
pkgdown::build_articles_index()
pkgdown::build_articles()
pkgdown::preview_site(path = "/articles")


# build -------------------------------------------------------------------
pkgdown::build_site(install=F)

pkgdown::deploy_to_branch()




##### NEW PKGDOWN WORKFLOW FOR DEPLOYING WITHOUT REBUILDING ARTICLES LOCALLY ####

# 1. SETUP: Fetch remote branches and create the worktree if missing
system("git fetch origin")

if (!dir.exists("docs")) {
  message("Creating 'docs' worktree linked to gh-pages...")
  system("git worktree add docs gh-pages")
} else {
  message("'docs' folder already exists. Skipping worktree setup.")
}

# 2. DEFINE THE DEPLOY FUNCTION
deploy_lazy <- function(message = "Update site") {
  # Check if docs folder exists
  if (!dir.exists("docs")) stop("The 'docs' folder is missing. Did the worktree setup fail?")
  
  message("Building site (Lazy)...")
  # This updates the files inside 'docs/'
  # It skips articles because it sees the existing HTML files
  pkgdown::build_site(lazy = TRUE)
  
  message("Deploying to gh-pages...")
  # The -C flag runs the git command inside the 'docs' directory
  system("git -C docs add .")
  
  # Suppress warnings in case there is 'nothing to commit'
  suppressWarnings(
    system(sprintf('git -C docs commit -m "%s"', message))
  )
  
  system("git -C docs push origin gh-pages")
  message("Deployment Complete!")
}

# 3. RUN IT
deploy_lazy()
