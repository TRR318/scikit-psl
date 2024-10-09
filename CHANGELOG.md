# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## 0.7.1 - 2024-10-09

### Fixed

- PSL uses stricter test to detect binary features


## 0.7.0 - 2024-03-27

### Added

- PSL classifier
    - probability predictions can return confidence intervals
    - probability calibration using BetaCalibration
    - stages can now be sliced and iterated (\__getitem\__(), \__iter\__())
- Metrics
    - Weighted loss metric
    - Rank loss metric

### Fixed

- PSL more robust against non-standard class labels like "True"/"False" instead of boolean values

## 0.6.3 - 2024-03-08

### Added

- PSL supports Dataframes as inputs

## 0.6.2 - 2024-02-01

### Added

- PSL supports instance weights

## 0.6.1 - 2023-12-15

### Added

- Extended precision at recall function

## 0.6.0 - 2023-12-07

### Added

- Significantly extended the configuration capabilities with predefined features to limit the PSLs searchspace

### Changed

- PSL global loss defaults to sum(cascade)
- rewrote/extracted expected entropy calculation

### Fixed

- PSL inspect is now more robust

## 0.5.1 - 2023-11-24

### Fixed

- PSL classifier optimization regarding global loss was incorrect

## 0.5.0 - 2023-11-16

### Added

- _ClassifierAtK
    - Sigmoid calibration additional to isotonic
- PSL classifier
    - Make optimization loss configurable
    - Small `searchspace_analyisis(·)` function makes lookahead choice more informed

### Fixed

- Fixed lookahead search space and considering global loss for model-sequence evaluation

### Changed

- Updated dependencies and added black
- Moved Binarizer to different module
- Moved PSL hyperparameters to constructor

## 0.4.2 - 2023-11-09

### Fixed

- _ClassifierAtK
    - Expected entropy for stage 0 now also calculated wrt. base 2
    - Data with only 0 or 1 is now also interpret as binary data

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
