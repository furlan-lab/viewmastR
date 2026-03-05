#!/usr/bin/env Rscript

# Filter a gene list to those annotated as cell surface proteins.

suppressPackageStartupMessages({
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
})

# Define required packages
bioc_pkgs <- c("AnnotationDbi", "GO.db", "org.Hs.eg.db", "org.Mm.eg.db")
cran_pkgs <- c("dplyr", "tibble") # Removed stringr as it wasn't strictly used

# Install missing packages
for (p in bioc_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) BiocManager::install(p, ask = FALSE, update = FALSE)
}
for (p in cran_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

suppressPackageStartupMessages({
  library(AnnotationDbi)
  library(GO.db)
  library(dplyr)
  library(tibble)
  library(org.Hs.eg.db)
  library(org.Mm.eg.db)
})

# ---- Core helpers ----

get_orgdb <- function(species = c("human", "mouse")) {
  species <- match.arg(species)
  if (species == "human") return(org.Hs.eg.db)
  if (species == "mouse") return(org.Mm.eg.db)
}

expand_go_cc <- function(go_ids) {
  go_ids <- unique(as.character(go_ids))
  go_ids <- go_ids[!is.na(go_ids) & go_ids != ""]
  
  # Robustly get offspring
  valid_keys <- go_ids[go_ids %in% keys(GOCCOFFSPRING)]
  
  if (length(valid_keys) > 0) {
    offspring_list <- as.list(GOCCOFFSPRING[valid_keys])
    offspring <- unname(unlist(offspring_list))
    all_terms <- unique(c(go_ids, offspring))
  } else {
    all_terms <- go_ids
  }
  return(all_terms)
}

# Main function
filter_membrane_genes <- function(
  gene_symbols,
  species = c("human", "mouse"),
  mode = c("surface_protein", "plasma_membrane_tm", "plasma_membrane", "any_membrane"),
  return_table = TRUE
) {
  species <- match.arg(species)
  mode <- match.arg(mode)
  
  orgdb <- get_orgdb(species)
  
  # ---- GO Term Definitions ----
  # surface_protein: The "Goldilocks" set for surface markers. 
  # Combines:
  #   - GO:0005886 (Plasma Membrane) - The most common annotation for surface proteins
  #   - GO:0009986 (Cell Surface)
  #   - GO:0005887 (Integral component of plasma membrane)
  #   - GO:0009897 (External side of plasma membrane)
  
  root_terms <- switch(
    mode,
    surface_protein    = c("GO:0005886", "GO:0009986", "GO:0005887", "GO:0009897"),
    plasma_membrane_tm = c("GO:0005887"),
    plasma_membrane    = c("GO:0005886"),
    any_membrane       = c("GO:0016020")
  )
  
  message(sprintf("Expanding GO terms for mode '%s'...", mode))
  term_set <- expand_go_cc(root_terms)
  message(sprintf("Found %d related GO terms.", length(term_set)))
  
  gene_symbols <- unique(as.character(gene_symbols))
  gene_symbols <- gene_symbols[!is.na(gene_symbols) & gene_symbols != ""]
  
  # 1. Map SYMBOL -> ENTREZID
  sym2ent <- AnnotationDbi::select(
    orgdb,
    keys = gene_symbols,
    keytype = "SYMBOL",
    columns = c("ENTREZID")
  ) %>%
    as_tibble() %>%
    filter(!is.na(ENTREZID))
  
  if (nrow(sym2ent) == 0) {
    warning("No SYMBOLs could be mapped to ENTREZID. Returning empty result.")
    return(list(kept_symbols = character(0), table = tibble()))
  }
  
  # 2. Get GO annotations (CC only)
  go_annot <- AnnotationDbi::select(
    orgdb,
    keys = unique(sym2ent$ENTREZID),
    keytype = "ENTREZID",
    columns = c("GO", "ONTOLOGY")
  ) %>%
    as_tibble() %>%
    filter(ONTOLOGY == "CC")
  
  # 3. Filter for hits
  hits <- go_annot %>%
    filter(GO %in% term_set) %>%
    distinct(ENTREZID)
  
  # 4. Merge back
  keep_tbl <- sym2ent %>%
    distinct(SYMBOL, ENTREZID) %>%
    mutate(is_membrane = ENTREZID %in% hits$ENTREZID)
  
  keep_symbols <- keep_tbl %>%
    filter(is_membrane) %>%
    pull(SYMBOL) %>%
    unique()
  
  if (!return_table) return(keep_symbols)
  
  # 5. Add Readable GO Names
  matched_go <- go_annot %>%
    filter(GO %in% term_set) %>%
    dplyr::select(ENTREZID, GO) %>% 
    distinct()
    
  if(nrow(matched_go) > 0) {
      go_definitions <- AnnotationDbi::select(
          GO.db, 
          keys = unique(matched_go$GO), 
          columns = "TERM", 
          keytype = "GOID"
      )
      
      matched_go <- matched_go %>%
          left_join(go_definitions, by=c("GO"="GOID")) %>%
          group_by(ENTREZID) %>%
          summarise(
              go_ids = paste(unique(GO), collapse = "; "),
              go_terms = paste(unique(TERM), collapse = "; "),
              .groups = "drop"
          )
  } else {
      matched_go <- tibble(ENTREZID=character(), go_ids=character(), go_terms=character())
  }

  out <- keep_tbl %>%
    left_join(matched_go, by = "ENTREZID") %>%
    arrange(desc(is_membrane), SYMBOL)
  
  list(
    kept_symbols = keep_symbols,
    table = out,
    mode = mode,
    species = species
  )
}

# ---- Test with CLEC2A ----
# res <- filter_membrane_genes("CLEC2A", species = "human", mode = "surface_protein")
# print(res$table)

extract_leftovers <- function(result_em, signatures, bulk_counts, thresh = 0.4){
  cell_exposures <- result_em$exposures[!rownames(result_em$exposures) %in% "Intercept",]
  noise_exposures <- result_em$exposures[rownames(result_em$exposures) %in% "Intercept", , drop=FALSE]
  # 2. Reconstruct the "Healthy Profile" for each patient
  # Matrix Math: [Genes x CellTypes] * [CellTypes x Samples]
  # Result: [Genes x Samples] representing only the healthy contribution
  healthy_profile <- signatures %*% cell_exposures
  residuals <- bulk_counts - healthy_profile
  residuals[residuals < 0] <- 0  # Clamp negative values to 0

  # 4. Normalize for library size (optional but recommended)
  # This gives you "CPM" of the tumor component
  residuals_cpm <- sweep(residuals, 2, colSums(bulk_counts), "/") * 1e6
  # Average residual expression across all samples
  avg_tumor_program <- rowMeans(residuals_cpm)

  # Top 20 potential targets
  avg_tumor_program <- sort(avg_tumor_program, decreasing = TRUE)

  # Correlate every gene's residual with the "Intercept Amount" vector
  # If a gene is part of the hidden program, it should track with the intercept usage
  correlations <- apply(residuals_cpm, 1, function(gene_vec) {
    cor(gene_vec, noise_exposures[1,])
  })

  # Filter for genes that are highly correlated AND highly expressed
  target_candidates <- data.frame(
    Gene = names(avg_tumor_program),
    Expression = avg_tumor_program,
    Correlation = correlations
  )

  out <- target_candidates[!is.na(target_candidates$Correlation),]
  out[out$Correlation > thresh,]
}



# ---- Interactive Testing Example ----
# Uncomment lines below to test immediately
# test_genes <- c("CD3D", "PTPRC", "GAPDH", "ACTB", "EGFR")
# result <- filter_membrane_genes(test_genes, species = "human", mode = "plasma_membrane")
# print(result$table)




#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(tibble)
})

# ---- Configuration ----
# We use the 'RNA consensus' dataset which combines HPA and GTEx data for robust nTPM values.
HPA_URL <- "https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip"

#https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip
#HPA_URL <- "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
CACHE_DIR <- "reference_data"
CACHE_FILE <- file.path(CACHE_DIR, "hpa_rna_consensus.tsv")

# ---- Helper: Manage Data Cache ----
ensure_hpa_data <- function() {
  if (!dir.exists(CACHE_DIR)) dir.create(CACHE_DIR)
  
  if (!file.exists(CACHE_FILE)) {
    message("Downloading Human Protein Atlas consensus data (approx. 15MB)...")
    
    # Download to a temporary zip file
    temp_zip <- tempfile(fileext = ".zip")
    download.file(HPA_URL, temp_zip, mode = "wb")
    
    # Unzip specifically the tsv we need
    message("Unzipping data...")
    unzip(temp_zip, exdir = CACHE_DIR)
    
    # HPA zip usually contains a file named 'rna_tissue_consensus.tsv'
    # We rename it to be safe or just identify it
    extracted_files <- list.files(CACHE_DIR, pattern = "rna_tissue_consensus.tsv", full.names = TRUE)
    if (length(extracted_files) > 0) {
      file.rename(extracted_files[1], CACHE_FILE)
    } else {
      stop("Download failed or unexpected zip structure.")
    }
    unlink(temp_zip)
    message("Database cached successfully.")
  }
  
  return(CACHE_FILE)
}

# ---- Main Function: Score Healthy Expression ----
score_healthy_expression <- function(gene_symbol, threshold_tpm = 1.0) {
  
  # 1. Load Data
  db_path <- ensure_hpa_data()
  
  # Read only the necessary columns to save memory if the file is huge
  # The file structure is: Gene, Gene name, Tissue, nTPM
  hpa_data <- read_tsv(
    db_path, 
    col_types = cols(
      Gene = col_character(),
      `Gene name` = col_character(),
      Tissue = col_character(),
      nTPM = col_double()
    ),
    progress = FALSE
  )
  
  # 2. Filter for the gene
  gene_data <- hpa_data %>%
    filter(`Gene name` %in% gene_symbol)
  
  if (nrow(gene_data) == 0) {
    warning(paste("Gene not found in HPA database:", gene_symbol))
    return(NULL)
  }
  
  # 3. Calculate Scores
  # We look for tissues above the threshold
  expressed_tissues <- gene_data %>%
    filter(nTPM >= threshold_tpm) %>%
    arrange(desc(nTPM))
  
  # Summary Statistics
  max_tissue <- gene_data %>% arrange(desc(nTPM)) %>% slice(1)
  total_tpm <- sum(gene_data$nTPM)
  
  # specific_score (Tau-like): 1 means highly specific to one tissue, 0 means ubiquitous
  # Simple calc: Max Expression / Total Expression (quick proxy for specificity)
  specificity_score <- if(total_tpm > 0) max_tissue$nTPM / total_tpm else 0
  
  # 4. Critical Organ Check (Safety Filter)
  # Define a list of "High Risk" organs where off-target toxicity is fatal
  critical_organs <- c("heart", "lung", "liver", "kidney", "brain", "pancreas")
  
  risk_hits <- expressed_tissues %>%
    filter(Tissue %in% critical_organs)
  
  is_safe_candidate <- nrow(risk_hits) == 0
  
  # 5. Output List
  list(
    gene = gene_symbol,
    is_safe_candidate = is_safe_candidate,
    specificity_score = round(specificity_score, 3), # Higher is better for targets
    max_expression_tissue = max_tissue$Tissue,
    max_expression_val = max_tissue$nTPM,
    critical_hits = risk_hits, # Tissues that might kill the patient
    full_profile = expressed_tissues # Full list of what lights up
  )
}

# ---- Example Usage ----
# Load one gene to test
# result <- score_healthy_expression("CD19")
# 
# print(paste("Target:", result$gene))
# print(paste("Safe Candidate (No Vital Organs > 1 TPM):", result$is_safe_candidate))
# print("Critical Organ Expression:")
# print(result$critical_hits)