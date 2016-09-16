# This script reads in both training data about passengers on the Titanic and tries to predict if passengers from a test dataset would have been likely to survive.

# Trying to recreate the R code from https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

using DataFrames
using DecisionTree

showln(x) = (show(x); println())

# Load in the training data
train = readtable("../input/train.csv")

println("The columns in the train set are:\n")
println(showcols(train))
println(@sprintf("\nThere are %d rows in the training set", nrow(train)))

# Load in the test data
test = readtable("../input/test.csv")

# Add a new column parsing out the passenger's formal title, if applicable
# Personal note: the final param in 'replace' needs to be dbl-quotes, otherwise "Invalid Character Literal" error pops up
train[:Title] = map((x) -> replace(x, r"(.*, )|(\..*)", ""), train[:Name])

# create a table showing counts of formal titles by gender
showln( by(train, [:Sex, :Title], nrow) )
