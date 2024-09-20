# quick testing -----------------------------------------------------------
pkgdown::clean_site(pkg = ".")
pkgdown::init_site(pkg = ".")
pkgdown::build_home_index()
pkgdown::preview_page("index.html")
pkgdown::build_article(name = "HowTo")
pkgdown::preview_page("articles/HowTo.html")

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

