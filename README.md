# Homework 2. Linear Regression

## Model
- Train linear regression model with regularisation on dataset, which would be sent you.
- Use structure from https://github.com/KamilyaKharisova/Ml_lib_students
- You need to write code, that calculates Moore-Penrose inverse with regularisation coefficient using SVD. You can use np.linalg.svd but don't use np.linalg.pinv.
- You need to write code for calculating weights for the model using Moore-Penrose inverse matrix and targets values. Don't use loops
-Basis function for this work are - polynomials

## Dataset
- You need to divide dataset on train(80%) validation(10%) and test(10%) sets. Permute data 

## Validation
- Validate max polynomial degree and regularisation coefficient
- Use Radom Search method, described in the presentation
- Train at leat 100 models

## Plots
- You need to find 10 best model according to error on validation set. Make plot with 10 points, where x-axis name of the model (max degree + regularisation coefficient) , y-axis error on valid set. Add to hover_data error on test set

- You need to make plots with two traces - model prediction and target values. Target trace need to be in markers mode,   model prediction - line. You need to make 2 plots for 2 models with (lambda=1e-5) and without (lambda=0) regularisation max degree of polynomials 100
