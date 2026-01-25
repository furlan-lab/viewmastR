# Extract the Shape of Every Tensor in a *burn* `.mpk` Checkpoint

Walks the arbitrary-depth module tree inside a **burn**‐generated
MessagePack checkpoint (`*.mpk`) and returns a tidy data frame listing
each tensor (e.g., `weight`, `bias`) and its dimensions. Handles any
number of linear (or other) layers and preserves nested sub-module names
via a dot-separated path.

## Usage

``` r
extract_mpk_shapes(file)
```

## Arguments

- file:

  name of model.mpk file
