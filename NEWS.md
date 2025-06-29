# viewmastR v0.3.0

* bug - fixed a bug that failed to correctly perform ReLu on all NN
* enhancement - upgrade to burn 0.17.1 and rust 2024 version
* enhancement - store feature names alongside model.mpk in file meta.mpk

# viewmastR v0.2.4

* bug - incorrectly printed training accuracy in specific situations
* bug - fixed an error in probability calculations
* performance improvement - enabled train_only mode so that inference can be done on large datasets separately; embedded minor performance enhancements throughout - See BigQuery Vignette

# viewmastR v0.2.3

* new feature - naive bayes implementation
* new feature - make3file function for easy exporting Seurat objects
* performance improvement - improved sparse variance function efficiency
* minor bug - fixed return type 'matrix' in training function
* minor bug - fixed tf_idf function
* enhancement - added debug function that enables tracking dimensions across learning steps
* enhancement - made a number of enhancements in rust code

# viewmastR v0.2.2
* minor changes not packaged as an actual release

# viewmastR v0.2.1

## New features

* use saved models 
* minor bugs

# viewmastR v0.1.5

## New features

* splat_bulk_reference  (released as viewmastRust)

# viewmastR v0.1.4

## New features

* sfc color function  (released as viewmastRust)

## Minor bug fixes and improvements

* many  (released as viewmastRust)

# viewmastR v0.1.2

## New features

* augment function  (released as viewmastRust)

## Minor bug fixes and improvements

* many  (released as viewmastRust)

# viewmastR v0.1.0

* first stable version (released as viewmastRust)
