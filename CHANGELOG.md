# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.4.1 - 2023-10-19

### Fixed

- Small import error

## 0.4.0 - 2023-10-17

### Added

- Add brute force threshold optimization method to find the global optimum, bisect optimizer remains default method

### Changed

- Restructured source files

## 0.3.1 - 2023-09-12

### Fixed

- PSL is now correctly handles when all instances belong to the negative class
- [#1](../../issues/1) if the first feature is assigned a negative score, it is now assigned the most negative score

## 0.3.0 - 2023-08-10

### Added

- PSL classifier can now run with continuous data and optimally (wrt. expected entropy) select thresholds to binarize
  the data

### Changed

- Significantly improved optimum calculation for MinEntropyBinarizer (the same optimization algorithm is shared with the
  psls internal binarization algorithm)

## 0.2.0 - 2023-08-10

### Added

- PSL classifier
    - introduced parallelization
    - implemented l-step lookahead
    - simple inspect(·) method that creates a tabular representation of the model

## 0.1.0 - 2023-08-08

### Added

- Initial implementation of the PSL algorithm
