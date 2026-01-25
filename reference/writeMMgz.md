# Write a compressed MatrixMarket file

This function writes a sparse matrix in the MatrixMarket format to a
compressed `.gz` file. The function handles both real and integer matrix
types.

## Usage

``` r
writeMMgz(x, file)
```

## Arguments

- x:

  A sparse matrix (typically a `dgCMatrix` or `ngCMatrix` object).

- file:

  A character string specifying the output file name, which will be
  compressed into `.gz` format.

## Value

This function does not return a value. It writes a file as a side
effect.

## Details

This function writes the matrix in the MatrixMarket coordinate format.
It first writes the header indicating the matrix type and size, and then
appends the matrix data. If the matrix is an `ngCMatrix`, it is treated
as an integer matrix, otherwise as a real matrix. The function
compresses the output into a `.gz` file.

## Examples

``` r
if (FALSE) { # \dontrun{
library(Matrix)
m <- Matrix(c(0, 1, 0, 2), 2, 2, sparse = TRUE)
writeMMgz(m, "matrix.mtx.gz")
} # }
```
