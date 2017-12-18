# AM 205 Final Project - Options Pricing

The aim of this project is to explore the use of the local volatility model
toward options pricing in a numerically stable way given the complication of
discreteness in real world data.

## Project Team

- [Francisco Rivera](mailto:frivera@college.harvard.edu)
- [Jiafeng (Kevin) Chen](mailto:jiafengchen@college.harvard.edu)

## Navigating the Code Base 

1. The TeX source and accompanying files for the report can be found in the
   `/report/` directory, as well as the [final PDF report](report/report.pdf)

2. [Report Figures.ipynb](Report%20Figure.ipynb) makes illustrative diagrams
   for the report (only graphical aids, not as reports of results or data).

3. [Monte Carlo Pricer.ipynb](Monte%20Carlo%20Pricer.ipynb) holds most of the
   code associated with pricing options given a local volatility function, as
   well as the tests on the accuracy of the Monte Carlo pricings.

4. [Finite Differences.ipynb](Finite%20Differences.ipynb) explores the finite
   differences approach to fitting the local volatility function.

5. [Results.ipynb](Results.ipynb) puts it all together and houses all of the
   major results that appear in the report.

6. The `lab_notebooks` directory houses intermediate work that has been
   refactored away.

7. The `report/scripts` directory includes the most relevant functions that were
   defined for the project, as well as the `.py` files in the main directory
   (i.e. [interpolated_local_vol.py](interpolated_local_vol.py) and
   [sample_prices](sample_prices.py)).



