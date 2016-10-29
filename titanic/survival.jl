# This script reads in both training data about passengers on the Titanic and tries to predict if passengers from a test dataset would have been likely to survive.

# Trying to recreate the R code from https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

module Survival

using DataFrames
using DecisionTree
using StatPlots

# Dabbled with UnicodePlots but I guess my terminal settings were conflicting
plotly()

showln(x) = (show(x); println())    # Quick function to show variable on its own line

# Load in the training data
train = readtable("../input/train.csv")

println("The columns in the train set are:\n")
println(showcols(train))
println(@sprintf("\nThere are %d rows in the training set", nrow(train)))

# Load in the test data
test = readtable("../input/test.csv")

# Attempt to combine 'train' and 'test' data frames (similar to R dplyr.bind_rows)
full = DataFrame(train)
full = vcat(full, test)
# Apparently could have also wrote 'full = [train ; test] as a shortcut to vcat

###
# Feature Engineering - Breaking down passenger names
###

# Add a new column parsing out the passenger's formal title, if applicable
# Personal note: the final param in 'replace' needs to be dbl-quotes, otherwise "Invalid Character Literal" error pops up
full[:Title] = map(x -> replace(x, r"(.*, )|(\..*)", ""), full[:Name])

# create a table showing counts of formal titles by gender
showln( by(full, [:Sex, :Title], nrow) )

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
    , full[:Title])

showln( by(full, [:Sex, :Title], nrow) )

# Determine surnames from passenger names
full[:Surname] = map(x -> split(x, r"[,.]")[1], full[:Name])

# Probaby could have gotten the unique surnames a better way
# Also didn't feel like looking up how to embed HTML tags, like the example
println(@sprintf("\nWe have %d unique surnames.", nrow(by(full, :Surname, nrow)) ))

###
# Feature Engineering - Does family size correlate with survival?
###

# Create a family size variable including the passenger themselves
full[:Fsize] = map((x,y) -> x + y + 1, full[:SibSp], full[:Parch])

# Create a family variable (to distinguish multiple families with same surname)
full[:Family] = map(x -> join(x, "_"), zip(full[:Surname], full[:Fsize]))

# Count the number of survivors per family size (from training set) and plot
train_counts =  by(full[1:891,:] , [:Fsize, :Survived], nrow)

p = bar(train_counts, :Fsize, :x1, group=:Survived)
xaxis!("Family Size")
xticks!(collect(1:11))
yaxis!("Count")
title!("Survival by Family Size")
#gui()

# Categgorize family sizes (1, 2-4, 5+)
full[:FsizeD] = map(x ->
    if x == 1
        "singleton"
    elseif x > 4
        "large"
    else    # x is > 1 and < 5
        "small"
    end
    , full[:Fsize])

### Currently unsure on how to create mosaic plot, so we'll just skip this

###
# Feature Engineering - Cabin Information
###

# Just showing we have a lot of missing cabin information
showln(full[:Cabin][1:28])

# Split the second entry per letter
showln( split(full[:Cabin][2], "") )

# Create a Deck designation using the first letter of the Cabin
full[:Deck] = map(x ->
    if isna(x)
        x
    else
        split(x, "")[1]
    end
    , full[:Cabin])
# According to earlier, 687 rows are missing cabin information, so it's not terribly useful

###
# Missingness - Adding sensible data for missing values
###

# Show that these are the 2 rows with missing embarkation data
# NOTE: Brackets around :Embarked returns DataFrame. No brackets returns String Array
showln(full[ [62, 830],[:Embarked] ])

println(@sprintf("We will infer their values for **embarkment** based on present data that we can imagine may be relevant: **passenger class** and **fare**. We see that they paid \$%d and \$%d respectively and their classes are %s and %s. So from where did they embark?",
    full[ [62, 830],[:Fare] ][1][1],
    full[ [62, 830],[:Fare] ][1][2],
    full[ [62, 830],[:Pclass] ][1][1],
    full[ [62, 830],[:Pclass] ][1][2],
    ))

# Create dataframe of only the passengers with embark information
# Personal note... the 'find' command is collecting indexes from rows where isna() returned 0
embarked = full[find(!isna(full[:,:Embarked])), :]
# Tutorial doesn't do this but one "Fare" is missing, and Julia doesn't like to plot with NA columns
embarked = embarked[find(!isna(embarked[:,:Fare])), :]
showcols(embarked)

# Boxplot of embarking location vs fare with respect to passenger classes
# NOTE: As of right now, I'm not sure how to "unstack" the boxplot
p2 = boxplot(embarked, :Embarked, :Fare, group=:Pclass)
hline!([80], ls=:dash, lc=:red, lw=2, label="")
yaxis!(y -> @sprintf("\$%d",y))
#gui()

# Passengers 62 and 830 apparently embarked from Charbourg ('C') so fix N/A
full[ [62,830], :Embarked ] = "C";

# Fix the null Fare value for this passenger (was eliminated from "embarked" subset)
showln(full[1044,:])
#Remove our NA fare passenger in this subset
fare = full[find(!isna(full[:,:Fare])), :]
S_3_fare = fare[ (fare[:Pclass] .== 3) & (fare[:Embarked] .== "S"), :]
# NOTE - Adding "." in front of the comparison operator allows for element-wise comparisons in arrays
# Also, one must use "&" instead of "&&" to combine conditions

# Density plot showing the fare for all 3rd class passengers boarding from Southampton
p3 = density(S_3_fare, :Fare, fc=:blue, fa=0.4, label="")
vline!([median(S_3_fare[:Fare])], ls=:dash, lc=:red, lw=1, label="")
xaxis!(x -> @sprintf("\$%d",x))
gui()

quit()


### Skipping to section 4 ###


###
# Prediction
###

### Split back into training and test datasets
# NOTE I never joined the datasets to begin with.  Eventually will edit code to do that.

### Building the RandomForests model

#Survival is our factor
# labels = convert(Array, train[:, :Survived])
# # These others are features we want to compare against (leaving out Child and Mother for now)
# features = convert(Array, train[:, [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked, :Title, :FsizeD] ])
# # Build the model, using 3 features per split (sqrt of total features), 100 trees, and 1.0 subsampling ratio
# rf_model = build_forest(labels, features, 3, 100, 1.0)
# apply_forest(rf_model, train)
#
# p = plot(rf_model, ylim=[0,0.36])

end # module Survival
