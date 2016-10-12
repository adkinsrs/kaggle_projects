# This script reads in both training data about passengers on the Titanic and tries to predict if passengers from a test dataset would have been likely to survive.

# Trying to recreate the R code from https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

module Survival

using DataFrames
using DecisionTree
using Gadfly

showln(x) = (show(x); println())    # Quick function to show variable on its own line

# Load in the training data
train = readtable("../input/train.csv")

println("The columns in the train set are:\n")
println(showcols(train))
println(@sprintf("\nThere are %d rows in the training set", nrow(train)))

# Load in the test data
test = readtable("../input/test.csv")

# Attempt to combine 'train' and 'test' data frames (similar to R dplyr.bind_rows)
#full = join(train, test, on=:PassengerId)

###
# Feature Engineering - Breaking down passenger names
###

# Add a new column parsing out the passenger's formal title, if applicable
# Personal note: the final param in 'replace' needs to be dbl-quotes, otherwise "Invalid Character Literal" error pops up
train[:Title] = map(x -> replace(x, r"(.*, )|(\..*)", ""), train[:Name])

# create a table showing counts of formal titles by gender
showln( by(train, [:Sex, :Title], nrow) )

# Rare titles are kept in an array
rare_title = ["Dona", "Lady", "the Countess","Capt", "Col", "Don",
                "Dr", "Major", "Rev", "Sir", "Jonkheer"]

# Also rename Mlle, Ms, Mme, and rare titles using in-place mapping
map!((x) ->
    if x == "Mlle" || x == "Ms"
        x = "Miss"
    elseif x == "Mme"
        x = "Mrs"
    elseif in(x, rare_title)
        x = "Rare Title"
    else
        x = x   # Needed this, otherwise "Nothing" would show up for other titles
    end
    , train[:Title])

showln( by(train, [:Sex, :Title], nrow) )

# Determine surnames from passenger names
train[:Surname] = map(x -> split(x, r"[,.]")[1], train[:Name])

# Probaby could have gotten the unique surnames a better way
println(@sprintf("\nWe have %d unique surnames.", nrow(by(train, :Surname, nrow)) ))

###
# Feature Engineering - Breaking down passenger names
###

# Create a family size variable including the passenger themselves
train[:Fsize] = map((x,y) -> x + y + 1, train[:SibSp], train[:Parch])

# Create a family variable
train[:Family] = map(x -> join(x, "_"), zip(train[:Surname], train[:Fsize]))

train_counts =  by(train, [:Fsize, :Survived], nrow)
p = plot(train_counts, x=:Fsize, y=:x1, color=:Survived,
    Guide.xlabel("Family Size"), Guide.ylabel("Count"),
    Guide.xticks(ticks=collect(1:11)),
    Geom.bar(position=:dodge),
    Scale.color_discrete()
)
img = SVG("survival_by_family_size.svg", 6inch, 4inch)
draw(img, p)


end # module Survival
