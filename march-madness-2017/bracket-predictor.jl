# My attempt to model the March Madness brackets based on the past 4 seasons, and apply it to predicting the 2017 tournament

module BracketPredictor

using DataFrames
using DecisionTree

cd("/home/shaun/git/kaggle_projects/march-madness-2017/")

showln(x) = (show(x); println())    # Quick function to show variable on its own line

# Load in the team IDs
teams = readtable("./input/Teams.csv")
season_stats = readtable("./input/RegularSeasonDetailedResults.csv")
tourney_stats = readtable("./input/TourneyDetailedResults.csv")
seeds = readtable("./input/TourneySeeds.csv")
slots = readtable("./input/TourneySlots.csv")

teams = DataFrame(teams)
season_stats = DataFrame(season_stats)
tourney_stats = DataFrame(tourney_stats)
seeds = DataFrame(seeds)
slots = DataFrame(slots)

# Things to consider (scratchpad):
### 1) Do regular season statistics correlations predict victories?
##### 1a) One caveat is we do not know quality of play (major vs mid-major conference)
### 2) Do these translate to similar predictions in the tournament?
### 3) Establish an ELO ranking system potentially
##### 3a) Use end of season tournament seeds (seed #s 1-10) to assign early ELO to next season.
#####     Start at 1600 for 1-seed, decrement by 10 per seed increase, everyone else is 1500
##### 3b) Get stats-per-game for each team in that season, and use correlation and ELO to assign probability

### Creating a table of average stats per team per season ###

# Let's make sure I imported my data correctly
println("Columns in the seasons_stats (detailed) dataset:")
showcols(season_stats)
println(@sprintf("\nNumber of rows: %d\n", nrow(season_stats)))

# Create a DataFrame listing each team's season averages by season
# I'm not keeping score because the winning team will always have a better score

# First get stats of winning team
team_wins = by(season_stats, [:Season, :Wteam], nrow)
winning_stats = by(season_stats, [:Season, :Wteam],
    df -> DataFrame( colwise(mean, df[ 9:21 ]) ))

# Next get stats of the losing team
team_losses = by(season_stats, [:Season, :Lteam], nrow)
losing_stats = by(season_stats, [:Season, :Lteam],
    df -> DataFrame( colwise(mean, df[ 22:34 ]) ))

# Renaming columns to allow for vcat to join properly
names!(winning_stats, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
  :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF])
names!(losing_stats, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF])

# Multiply stats by wins/losses to give them weight for total averages
# NOTE: I tried to do a one-liner for this (I'm a Perl guy), but my inexperience with Julia prevented this
nrows, ncols = size(winning_stats)
for row in 1:nrows
  for col in 3:ncols
    winning_stats[row,col] *= team_wins[row, 3]
  end
end

nrows, ncols = size(losing_stats)
for row in 1:nrows
  for col in 3:ncols
    losing_stats[row,col] *= team_losses[row, 3]
  end
end

# Let's get the total sum of games played per team per season
rename!(team_wins, :Wteam, :Team)
rename!(team_losses, :Lteam, :Team)
total_team_games = join(team_wins, team_losses, on=[:Season, :Team], kind=:outer)
rename!(total_team_games, :x1, :Wins)
rename!(total_team_games, :x1_1, :Losses)

# Teams with no losses or no wins have NA for those values.  Need to replace with 0
total_team_games[:Wins] = map(x -> if isna(x) 0 else x end, total_team_games[:Wins])
total_team_games[:Losses] = map(x -> if isna(x) 0 else x end, total_team_games[:Losses])
total_team_games[:TotalGames] = total_team_games[:Wins] .+ total_team_games[:Losses]
sort!(total_team_games, cols=[order(:Season), order(:Team)])
#showln(total_team_games)

# Each team's seasonal statistics should be here now.
temp_comb_stats = vcat(winning_stats, losing_stats)
team_stats_by_season = aggregate(temp_comb_stats, [:Season, :Team], sum)

#Divide total stats by total games to get average
nrows, ncols = size(team_stats_by_season)
for row in 1:nrows
  for col in 3:ncols
    team_stats_by_season[row,col] /= total_team_games[row,:TotalGames]
  end
end
names!(team_stats_by_season, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF])
showln(team_stats_by_season)

### Creating a Training Set ###

# Now that I have average stats per team per season, I would like to convert the
# original season data into a training data set

# Couldn't figure out a shorthand way to do this so here's the long way.
# Lets get the winning team's data in first
season_training_data = DataFrame()
season_training_data[:Season] = season_stats[:Season]
season_training_data[:Team] = season_stats[:Wteam]
season_training_data[:FGM] = season_stats[9]
season_training_data[:FGA] = season_stats[10]
season_training_data[:FGM3] = season_stats[11]
season_training_data[:FGA3] = season_stats[12]
season_training_data[:FTM] = season_stats[13]
season_training_data[:FTA] = season_stats[14]
season_training_data[:OffR] = season_stats[15]
season_training_data[:DefR] = season_stats[16]
season_training_data[:Assist] = season_stats[17]
season_training_data[:TO] = season_stats[18]
season_training_data[:Steal] = season_stats[19]
season_training_data[:Block] = season_stats[20]
season_training_data[:PF] = season_stats[21]
season_training_data[:Win] = map(x -> x=1, season_training_data[:Season])

# For the losing team's data I will create a separate DataFrame
temp_df = DataFrame()
temp_df[:Season] = season_stats[:Season]
temp_df[:Team] = season_stats[:Lteam]
temp_df[:FGM] = season_stats[22]
temp_df[:FGA] = season_stats[23]
temp_df[:FGM3] = season_stats[24]
temp_df[:FGA3] = season_stats[25]
temp_df[:FTM] = season_stats[26]
temp_df[:FTA] = season_stats[27]
temp_df[:OffR] = season_stats[28]
temp_df[:DefR] = season_stats[29]
temp_df[:Assist] = season_stats[30]
temp_df[:TO] = season_stats[31]
temp_df[:Steal] = season_stats[32]
temp_df[:Block] = season_stats[33]
temp_df[:PF] = season_stats[34]
temp_df[:Win] = map(x -> x=0, temp_df[:Season])

# Combine both dataframes
season_training_data = vcat(season_training_data, temp_df)
showln(season_training_data)

# Take the converted training dataset and create a forest decision trees

#Random seed
srand(754)

# Win/Loss is our factor
labels = convert(Array, season_training_data[:, :Win])
# These others are features we want to compare against (leaving out Child and Mother for now)
features = convert(Array, season_training_data[:, [:FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF] ])
test_features = convert(Array, season_training_data[:, [:FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF] ])
# Build the model, using 3 features per split (sqrt of total features), 100 trees, and 1.0 subsampling ratio
rf_model = build_forest(labels, features, 3, 100, 1.0)

print_tree(rf_model, 2)
# # Apply our random forest model on the test dataset
# prediction = apply_forest(rf_model, test_features)
# # Create the solution dataframe that will be submitted
# solution = DataFrame(PassengerId = test[:PassengerId], Survived = prediction)
# writetable("titanic_rf_mod.csv", solution, header=true)

end # module BracketPredictor
