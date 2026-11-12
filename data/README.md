# Data access

The UCI source dataset is not included in this repository. Download the
Polish Companies Bankruptcy dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/365/polish%2Bcompanies%2Bbankruptcy%2Bdata)
and place the file named `5year.arff` at:

`data/raw/polish_companies_5year.arff`

The scripts convert the ARFF file to CSV and generate the processed splits
locally. Generated CSV files and the raw source file are excluded from Git.

The UCI dataset is licensed under CC BY 4.0. Cite:

Tomczak, S. (2016). *Polish Companies Bankruptcy* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5F600.
