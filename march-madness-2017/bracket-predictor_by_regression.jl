#!/usr/bin/env julia

# My attempt to model the March Madness brackets based on the past 4 seasons, and apply it to predicting the 2017 tournament

# This particular one uses GLM to perform regression analysis to predict the brackets
# Idea to explore regression was inspired from https://www.kaggle.com/ajniggles/march-machine-learning-mania-2017/logistic-regression-and-game-round-calculator


using DataFrames
using GLM

# cd("/home/shaun/git/kaggle_projects/march-madness-2017/")

showln(x) = (show(x); println())    # Quick function to show variable on its own line

season_stats = readtable("./input/RegularSeasonDetailedResults.csv")
tourney_stats = readtable("./input/TourneyDetailedResults.csv")

### Creating a table of average stats per team per season ###

# Create a DataFrame listing each team's season averages by season
# Also keep track of opponents season averages as well (FG defense, etc.)
# I'm not keeping score because the winning team will always have a better score

# First get stats of winning team
winning_stats = by(season_stats, [:Season, :Wteam],
    df -> DataFrame( df[ 9:34 ] ))

# Next get stats of the losing team
losing_stats = by(season_stats, [:Season, :Lteam],
    df -> DataFrame( df[ 9:34 ] ))

# Renaming columns to allow for vcat to join properly
names!(winning_stats, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
  :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF, :oFGM, :oFGA, :oFGM3, :oFGA3, :oFTM, :oFTA,
    :oOffR, :oDefR, :oAssist, :oTO, :oSteal, :oBlock, :oPF])
names!(losing_stats, [:Season, :Team, :oFGM, :oFGA, :oFGM3, :oFGA3, :oFTM, :oFTA,
  :oOffR, :oDefR, :oAssist, :oTO, :oSteal, :oBlock, :oPF, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF])

# Create new columns for wins
winning_stats[:Win] = map(x -> x=1, winning_stats[:Season])
losing_stats[:Win] = map(x -> x=0, losing_stats[:Season])

combined_stats = vcat(winning_stats, losing_stats)
team_stats_by_season = aggregate(combined_stats, [:Season, :Team], mean)
names!(team_stats_by_season, [:Season, :Team, :oFGM, :oFGA, :oFGM3, :oFGA3, :oFTM, :oFTA,
  :oOffR, :oDefR, :oAssist, :oTO, :oSteal, :oBlock, :oPF, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF, :Win])
#showln(team_stats_by_season)

# Add some percentage information
team_stats_by_season[:FG_pct] = map((x,y) -> x / y, team_stats_by_season[:FGM], team_stats_by_season[:FGA])
team_stats_by_season[:oFG_pct] = map((x,y) -> x / y, team_stats_by_season[:oFGM], team_stats_by_season[:oFGA])
team_stats_by_season[:FG3_pct] = map((x,y) -> x / y, team_stats_by_season[:FGM3], team_stats_by_season[:FGA3])
team_stats_by_season[:oFG3_pct] = map((x,y) -> x / y, team_stats_by_season[:oFGM3], team_stats_by_season[:oFGA3])
team_stats_by_season[:FT_pct] = map((x,y) -> x / y, team_stats_by_season[:FTM], team_stats_by_season[:FTA]) # no opponent data... can't influence that
team_stats_by_season[:Tot_Reb] = map((x,y) -> x + y, team_stats_by_season[:OffR], team_stats_by_season[:DefR])
team_stats_by_season[:oTot_Reb] = map((x,y) -> x + y, team_stats_by_season[:oOffR], team_stats_by_season[:oDefR])
team_stats_by_season[:Ast_TO_ratio] = map((x,y) -> x / y, team_stats_by_season[:Assist], team_stats_by_season[:TO])
team_stats_by_season[:oAst_TO_ratio] = map((x,y) -> x / y, team_stats_by_season[:oAssist], team_stats_by_season[:TO])
#showln(team_stats_by_season)

### Creating a Training Set ###
### Lets create a training dataset with tournament stats (since it's a smaller dataset)

# Had to convert Season year into a float so that Win type would not be "Any"
winning_tourney_stats = by(tourney_stats, [:Season, :Wteam],
    df -> DataFrame( colwise(mean, df[ 9:34 ]) ))
winning_tourney_stats[:Win] = map(x -> x=1., float(winning_tourney_stats[:Season]))

# For the losing team's data I will create a separate DataFrame
losing_tourney_stats = by(tourney_stats, [:Season, :Lteam],
    df -> DataFrame( colwise(mean, df[ 9:34 ]) ))
losing_tourney_stats[:Win] = map(x -> x=0., float(losing_tourney_stats[:Season]))

# Renaming columns to allow for vcat to join properly
names!(winning_tourney_stats, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
  :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF, :oFGM, :oFGA, :oFGM3, :oFGA3, :oFTM, :oFTA,
    :oOffR, :oDefR, :oAssist, :oTO, :oSteal, :oBlock, :oPF, :Win])
names!(losing_tourney_stats, [:Season, :Team, :oFGM, :oFGA, :oFGM3, :oFGA3, :oFTM, :oFTA,
  :oOffR, :oDefR, :oAssist, :oTO, :oSteal, :oBlock, :oPF, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF, :Win])

# Combine both dataframes
tourney_training_data = vcat(winning_tourney_stats, losing_tourney_stats)

# Add some percentage information
tourney_training_data[:FG_pct] = map((x,y) -> x / y, tourney_training_data[:FGM], tourney_training_data[:FGA])
tourney_training_data[:oFG_pct] = map((x,y) -> x / y, tourney_training_data[:oFGM], tourney_training_data[:oFGA])
tourney_training_data[:FG3_pct] = map((x,y) -> x / y, tourney_training_data[:FGM3], tourney_training_data[:FGA3])
tourney_training_data[:oFG3_pct] = map((x,y) -> x / y, tourney_training_data[:oFGM3], tourney_training_data[:oFGA3])
tourney_training_data[:FT_pct] = map((x,y) -> x / y, tourney_training_data[:FTM], tourney_training_data[:FTA]) # no opponent data... can't influence that
tourney_training_data[:Tot_Reb] = map((x,y) -> x + y, tourney_training_data[:OffR], tourney_training_data[:DefR])
tourney_training_data[:oTot_Reb] = map((x,y) -> x + y, tourney_training_data[:oOffR], tourney_training_data[:oDefR])
tourney_training_data[:Ast_TO_ratio] = map((x,y) -> x / y, tourney_training_data[:Assist], tourney_training_data[:TO])
tourney_training_data[:oAst_TO_ratio] = map((x,y) -> x / y, tourney_training_data[:oAssist], tourney_training_data[:TO])
# showln(tourney_training_data)

# Add seasonal average stats to tourney stats for a training model
train = vcat(tourney_training_data, team_stats_by_season)
test = train[ train[:Season] .>= 2013,: ]

### Time to do Regression ###

# Create the regression model
# To avoid overfitting, I'm only considering % and ratio stats
model = glm( Win ~ FG_pct + oFG_pct + FG3_pct + oFG3_pct + FT_pct + Tot_Reb + oTot_Reb + Ast_TO_ratio + oAst_TO_ratio,
    train, Binomial(), LogitLink() )
println(model)

prediction = predict(model, test)
test[:prediction] = map(x -> x, prediction)

# # Time to read in the sample submission CSV and make our own
# sample = readtable("./input/sample_submission.csv")
# for row in 1:nrow(sample)
#     # Split out season, team1, and team2
#     id = split(sample[row, :id], "_")
#     # Convert SubStrings to Int
#     season = parse(Int, id[1])
#     team1 = parse(Int, id[2])
#     team2 = parse(Int, id[3])
#     # Grab the win probability for each team
#     team1_win = team_win_probability[ (team_win_probability[:Season] .== season) & (team_win_probability[:Team] .== team1), :Win]
#     team2_win = team_win_probability[ (team_win_probability[:Season] .== season) & (team_win_probability[:Team] .== team2), :Win]
#     team1_win = team1_win[1,1]
#     team2_win = team2_win[1,1]
#     #showln(team1_win)
#     #showln(team2_win)
#     probability = team1_win - team2_win + 0.5
#     # Set upper and lower bounds
#     if probability < 0
#         probability = 0
#     elseif probability > 1
#         probability = 1
#     end
#     sample[row, :pred] = probability
# end
#
# # Write results to csv, and call it a day
# #showln(sample)
# writetable("submission_rf.csv", sample)
