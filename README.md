# Diamonds-Dataset
This is a project attempting to predict the price of diamonds using features of diamonds, such as cut clarity and colour. This project is a work in progress - improvements/alternative methods will be uploaded in the future. The project uses a deep-model - a fully connected neural network and ensemble methods such as random forests and gradient boosting. The project contains an exploratory data analysis (EDA), data cleaning, feature engineering, dimensionality reduction and the model building files.

The dataset was found on kaggle: https://www.kaggle.com/shivam2503/diamonds

The data contains:

- price: in US dollars (\$326--\$18,823)

- carat: weight of the diamond (0.2--5.01)

- cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)

- color: from J (worst) to D (best)

- clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

- x: length in mm (0--10.74)

- y: width in mm (0--58.9)

- z :depth in mm (0--31.8)

- depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

- table: width of top of diamond relative to widest point (43--95)
