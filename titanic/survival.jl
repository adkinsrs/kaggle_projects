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

# Formerly used nrow(by(full, :Surname, nrow)) but discovered how to pool DataArrays into factors
# Also didn't feel like looking up how to embed HTML tags, like the example
println(@sprintf("\nWe have %d unique surnames.", length(levels(pool(full[:Surname]))) ))

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
# NOTE: I comment out the gui() command for working plots, so I don't have to see them when testing later plots

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
# Personal note... also could have used full[complete_cases(full[:,[:Embarked]]), :]
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
#Remove our NA fare passenger in this subset    (next 2 lines do the same thing)
#fare = full[find(!isna(full[:,:Fare])), :]
fare = full[complete_cases( full[:, [:Fare]] ), :]

S_3_fare = fare[ (fare[:Pclass] .== 3) & (fare[:Embarked] .== "S"), :]
# NOTE - Adding "." in front of the comparison operator allows for element-wise comparisons in arrays
# Also, one must use "&" instead of "&&" to combine conditions

# Density plot showing the fare for all 3rd class passengers boarding from Southampton
p3 = density(S_3_fare, :Fare, fill=(0, 0.4, colorant"#99d6ff"), label="")
vline!([median(S_3_fare[:Fare])], ls=:dash, lc=:red, lw=1, label="")
xaxis!(x -> @sprintf("\$%d",x))
#gui()

# Set passenger 1044's fare equal to the median of the Southampton 3rd class fares ($8.05)
full[ 1044, :Fare ] = median(S_3_fare[:Fare])

###
# Missingness - Predicting missing age values based on a model
###

println( @sprintf("Number of missing age values is %d", length(find(isna( full[:Age] )))) )

# Turn these dataframe columns into factors
pool!(full, [:PassengerId, :Pclass, :Sex, :Embarked, :Title, :Surname, :Family, :FsizeD])

# I can't really find a Julia package related to rpart or mice.  I'm not a stats guy but there probably is one.
srand(129)

# Create an array of valid passenger ages
age_only = full[complete_cases( full[:, [:Age] ] ), :Age]
# Create a histogram of passenger age distribution
p4 = histogram(age_only, normalize=true, bins=16, fc=:darkgreen, ylims=(0,0.04), xticks=0:20:80)
histogram!(title="Age: Original Data", legend=:none, xlabel="Age", ylabel="Density")
#gui()

# Since I can't do regression yet, I can't show the mice graph, and thus apply output to Age column
# ... but I'll just fill in the NA ages with random ages from age_only for now just to have something

for row in 1:nrow(full)
    temp = full[row, :Age]
    if isna(temp)
        full[row,:Age] = rand(minimum(age_only):maximum(age_only))
    end
end

# Using the original training rows
# R breaks the subplots by sex in the plot code, but I have to split in advance
train = full[1:891,:]
male_full = train[train[:Sex] .== "male", :]
female_full = train[train[:Sex] .== "female", :]

# Two histograms showing male and female age and survivorship.
# I can't seem to split both datasets into seperate series from the same code line
p5 = histogram(male_full, :Age, group=:Survived, layout=2)
histogram!(female_full, :Age, group=:Survived, label="")
plot!(c=[:red :blue], ylims=(0,70), xticks=0:20:80)
#gui()

# Categorize passengers as either children or adults
full[:Child] = map(x ->
    if x < 18
        "Child"
    else
        "Adult"
    end
    , full[:Age])

showln(by(full, [:Child, :Survived], nrow))

# Categorizing mothers now
full[:Mother] = "Not Mother"
for row in 1:nrow(full)
    if (full[row, :Sex] == "female"
        && full[row, :Parch] > 0
        && full[row, :Age] > 18
        && full[row, :Title] != "Miss")
        full[row, :Mother] = "Mother"
    end
end

showln(by(full, [:Mother, :Survived], nrow))

# Categorize child and mother fields
pool!(full, [:Child, :Mother])

###
# Prediction - Split back into training and test datasets
###

train = full[1:891,:]
test = full[892:1309,:]

###
# Prediction - Building the RandomForests model
###

# Random seed
srand(754)

#Survival is our factor
labels = convert(Array, train[:, :Survived])
# These others are features we want to compare against (leaving out Child and Mother for now)
features = convert(Array, train[:, [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked, :Title, :FsizeD, :Child, :Mother] ])
test_features = convert(Array, test[:, [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked, :Title, :FsizeD, :Child, :Mother] ])
# Build the model, using 3 features per split (sqrt of total features), 100 trees, and 1.0 subsampling ratio
rf_model = build_forest(labels, features, 3, 100, 1.0)

###
# Prediction - Ranking variable importance
###

###
# Prediction - Time for the actual Prediction
###

# Apply our random forest model on the test dataset
prediction = apply_forest(rf_model, test_features)
# Create the solution dataframe that will be submitted
solution = DataFrame(PassengerId = test[:PassengerId], Survived = prediction)
writetable("titanic_rf_mod.csv", solution, header=false)


end # module Survival
