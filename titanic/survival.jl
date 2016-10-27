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
        "Miss"
    elseif x == "Mme"
        "Mrs"
    elseif in(x, rare_title)
        "Rare Title"
    else
        x   # Needed this, otherwise "Nothing" would show up for other titles
    end
    , train[:Title])

showln( by(train, [:Sex, :Title], nrow) )

# Determine surnames from passenger names
train[:Surname] = map(x -> split(x, r"[,.]")[1], train[:Name])

# Probaby could have gotten the unique surnames a better way
# Also didn't feel like looking up how to embed HTML tags, like the example
println(@sprintf("\nWe have %d unique surnames.", nrow(by(train, :Surname, nrow)) ))

###
# Feature Engineering - Does family size correlate with survival?
###

# Create a family size variable including the passenger themselves
train[:Fsize] = map((x,y) -> x + y + 1, train[:SibSp], train[:Parch])

# Create a family variable (to distinguish multiple families with same surname)
train[:Family] = map(x -> join(x, "_"), zip(train[:Surname], train[:Fsize]))

# Count the number of survivors per family size and plot
train_counts =  by(train, [:Fsize, :Survived], nrow)
p = plot(train_counts, x=:Fsize, y=:x1, color=:Survived,
    Guide.xlabel("Family Size"), Guide.ylabel("Count"),
    Guide.xticks(ticks=collect(1:11)),
    Geom.bar(position=:dodge),
    Scale.color_discrete()
)
img = SVG("survival_by_family_size.svg", 6inch, 4inch)
draw(img, p)

# Categgorize family sizes (1, 2-4, 5+)
train[:FsizeD] = map(x ->
    if x == 1
        "singleton"
    elseif x > 4
        "large"
    else    # x is > 1 and < 5
        "small"
    end
    , train[:Fsize])
#showln(train)

### Gadfly does not seem to support a mosaic plot, so we'll just skip this

###
# Feature Engineering - Cabin Information
###

# Just showing we have a lot of missing cabin information
showln(train[:Cabin][1:28])

# Split the second entry per letter
showln( split(train[:Cabin][2], "") )

# Create a Deck designation using the first letter of the Cabin
train[:Deck] = map(x ->
    if isna(x)
        x
    else
        split(x, "")[1]
    end
    , train[:Cabin])
# According to earlier, 687 rows are missing cabin information, so it's not terribly useful

###
# Missingness - Adding sensible data for missing values
###

# Show that these are the 2 rows with missing embarkation data
showln(train[ [62, 830], [:Embarked] ])

println(@sprintf("We will infer their values for **embarkment** based on present data that we can imagine may be relevant: **passenger class** and **fare**. We see that they paid \$%d and \$%d respectively and their classes are %s and %s. So from where did they embark?",
    train[ [62, 830], [:Fare] ][1][1],
    train[ [62, 830], [:Fare] ][1][2],
    train[ [62, 830], [:Pclass] ][1][1],
    train[ [62, 830], [:Pclass] ][1][2],
    ))

# Create dataframe of only the passengers with embark information
# Personal note... the 'find' command is collecting indexes from rows where isna() returned 0
embarked = train[find(!isna(train[:,:Embarked])), :]

# Boxplot of embarking location vs fare with respect to passenger classes
# NOTE - The example has a Geom.hline red color line, but I'm getting StackOverflow errors when adding that
p2 = plot(
    embarked, x=:Embarked, y=:Fare, color=:Pclass, yintercept=[80],
    Geom.boxplot(), Geom.hline(color=colorant"red", size=2mm),
    Scale.y_continuous(labels = y -> @sprintf("\$%d",y))
)
img2 = SVG("embark_fare_pclass.svg", 6inch, 4inch)
draw(img2, p2)

end # module Survival
