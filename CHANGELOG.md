# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.0 - 2023-08-10

### Added

- PSL classifier can now run with continuous data and optimally (wrt. expected entropy) select thresholds to binarize the data

### Changed

- Significantly improved optimum calculation for MinEntropyBinarizer (the same optimization algorithm is shared with the psls internal binarization algorithm)


## 0.2.0 - 2023-08-10

### Added

- PSL classifier
  - introduced parallelization
  - implemented l-step lookahead
  - simple inspect(·) method that creates a tabular representation of the model
    

## 0.1.0 - 2023-08-08

### Added

- Initial implementation of the PSL algorithm