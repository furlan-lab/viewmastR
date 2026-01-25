# Extract linear-layer weights and map them to feature / class names

Reads a model exported by the Rust Burn pipeline together with its
companion metadata file and returns a tidy **weight matrix** whose rows
correspond to the original feature names and whose columns correspond to
the class labels.

## Usage

``` r
get_weights(dir)
```

## Arguments

- dir:

  `character(1)` Path to the artefact directory that contains *both*
  files produced by `run_custom()`:

  - `model.mpk` – tensor weights saved via
    `NamedMpkFileRecorder::<FullPrecisionSettings>`

  - `meta.mpk` – MessagePack blob created by `save_artifacts()` holding
    `feature_names` and `class_labels`.

## Value

A base-`data.frame` with dimension \\(\\\\features) \times
(\\\\classes)\\, where `rownames(wmat)` are the feature names and
`colnames(wmat)` are the class labels. Cell *(i,j)* is the weight
connecting feature *i* to logit *j*.

## Details

Internally the function:

1.  deserialises the two MessagePack files with **msgpackR**;

2.  raw‐decodes the tensor bytes through `decode_param()`;

3.  reshapes the flat vector into a column-major matrix using the stored
    shape (`[out_dim, in_dim]`);

4.  transposes it so that rows align with features;

5.  re-labels rows and columns from the metadata lists.

The resulting object is ready for `pheatmap()`, `corrplot()`, or
[`as.matrix()`](https://rdrr.io/r/base/matrix.html) for further
analysis.

## See also

- `msgpack_read()` from **msgpackR** – generic MessagePack reader

- `decode_param()` – helper that converts Burn tensor blobs into R
  vectors

## Examples

``` r
if (FALSE) { # \dontrun{
w <- get_weights("artifacts/run-42")
head(w[, 1:5])         # first 5 classes

# visualise top positive / negative features for class 3
cls <- 3
w_sorted <- w[order(w[, cls]), cls]
barplot(tail(w_sorted, 10), horiz = TRUE, las = 1)
barplot(head(w_sorted, 10), horiz = TRUE, las = 1)
} # }
```
