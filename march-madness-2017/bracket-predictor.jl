# My attempt to model the March Madness brackets based on the past 4 seasons, and apply it to predicting the 2017 tournament
using DataFrames
using DecisionTree

# Things to consider (scratchpad):
### 1) Do regular season statistics correlations predict victories?
##### 1a) One caveat is we do not know quality of play (major vs mid-major conference)
### 2) Do these translate to similar predictions in the tournament?
### 3) Establish an ELO ranking system potentially
##### 3a) Use end of season tournament seeds (seed #s 1-10) to assign early ELO to next season.
#####     Start at 1600 for 1-seed, decrement by 10 per seed increase, everyone else is at end-of-season value
##### 3b) Get stats-per-game for each team in that season, and use correlation and ELO to assign probability
### 4) Create ratios (Assist-to-turnover, FG%, etc.)
### 5) Get opponent statistics per team (FG% defense, forced turnovers, etc.)

# cd("/home/shaun/git/kaggle_projects/march-madness-2017/")

showln(x) = (show(x); println())    # Quick function to show variable on its own line

# Load in the team IDs
teams = readtable("./input/Teams.csv")
season_stats = readtable("./input/RegularSeasonDetailedResults.csv")
tourney_stats = readtable("./input/TourneyDetailedResults.csv")
# seeds = readtable("./input/TourneySeeds.csv")
# slots = readtable("./input/TourneySlots.csv")

### Creating a table of average stats per team per season ###

# Let's make sure I imported my data correctly
# println("Columns in the seasons_stats (detailed) dataset:")
# showcols(season_stats)
# println(@sprintf("\nNumber of rows: %d\n", nrow(season_stats)))

# Create a DataFrame listing each team's season averages by season
# Also keep track of opponents season averages as well (FG defense, etc.)
# I'm not keeping score because the winning team will always have a better score

# First get stats of winning team
team_wins = by(season_stats, [:Season, :Wteam], nrow)
winning_stats = by(season_stats, [:Season, :Wteam],
    df -> DataFrame( colwise(mean, df[ 9:34 ]) ))

# Next get stats of the losing team
team_losses = by(season_stats, [:Season, :Lteam], nrow)
losing_stats = by(season_stats, [:Season, :Lteam],
    df -> DataFrame( colwise(mean, df[ 9:34 ]) ))

# Renaming columns to allow for vcat to join properly
names!(winning_stats, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
  :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF, :oFGM, :oFGA, :oFGM3, :oFGA3, :oFTM, :oFTA,
    :oOffR, :oDefR, :oAssist, :oTO, :oSteal, :oBlock, :oPF])
names!(losing_stats, [:Season, :Team, :oFGM, :oFGA, :oFGM3, :oFGA3, :oFTM, :oFTA,
  :oOffR, :oDefR, :oAssist, :oTO, :oSteal, :oBlock, :oPF, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
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

# Divide total stats by total games to get average
nrows, ncols = size(team_stats_by_season)
for row in 1:nrows
  for col in 3:ncols
    team_stats_by_season[row,col] /= total_team_games[row,:TotalGames]
  end
end
names!(team_stats_by_season, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
  :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF, :oFGM, :oFGA, :oFGM3, :oFGA3, :oFTM, :oFTA,
  :oOffR, :oDefR, :oAssist, :oTO, :oSteal, :oBlock, :oPF])

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

winning_tourney_stats = by(tourney_stats, [:Season, :Wteam],
    df -> DataFrame( colwise(mean, df[ 9:34 ]) ))
winning_tourney_stats[:Win] = map(x -> x=1, winning_tourney_stats[:Season])

# For the losing team's data I will create a separate DataFrame
losing_tourney_stats = by(tourney_stats, [:Season, :Lteam],
    df -> DataFrame( colwise(mean, df[ 9:34 ]) ))
losing_tourney_stats[:Win] = map(x -> x=0, losing_tourney_stats[:Season])

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

### Create a forest decision tree ###

#Random seed
srand(754)

# Some stats that I will leave out for consideration of a team's chances of winning:
### FGA, FGA3, FTA - More attempts lead to more scores but percentage stats accounts for this
### oFGA, oFGA3 - same reason as above
### oFTM, oFTA - cannot directly influence opponent's chances of making free throws (outside of crowd tauntings)
### Assist, oAssist - Assists only count due to a score, so use FGM.  Will use in Assist/TO ratio though
### Steal, oSteal - Steals always count as a turnover so use TO.

# Win/Loss is our factor
labels = convert(Array, tourney_training_data[:, :Win])
# Training data will be based on tournament games
features = convert(Array, tourney_training_data[:, [:FGM, :FGM3, :FTM, :OffR, :DefR, :TO, :Block, :PF, :oFGM, :oFGM3, :oOffR, :oDefR, :oTO, :oBlock, :oPF, :FG_pct, :oFG_pct, :FG3_pct, :oFG3_pct, :Tot_Reb, :oTot_Reb, :Ast_TO_ratio, :oAst_TO_ratio] ])
# Test data will be based on season averages
test_features = convert(Array, team_stats_by_season[:, [:FGM, :FGM3, :FTM, :OffR, :DefR, :TO, :Block, :PF, :oFGM, :oFGM3, :oOffR, :oDefR, :oTO, :oBlock, :oPF, :FG_pct, :oFG_pct, :FG3_pct, :oFG3_pct, :Tot_Reb, :oTot_Reb, :Ast_TO_ratio, :oAst_TO_ratio] ])
# Build the model, using 4 features per split (sqrt of total features), 100 trees, and 1.0 subsampling ratio
rf_model = build_forest(labels, features, 4, 100, 1.0)
# This applies the probability of winning a random game
win_probability = apply_forest_proba(rf_model, test_features, [1,0])
team_win_probability = DataFrame(Season = team_stats_by_season[:Season],
    Team = team_stats_by_season[:Team], Win = win_probability[:,1])
# showln(team_win_probability)

# Time to read in the sample submission CSV and make our own
sample = readtable("./input/sample_submission.csv")
for row in 1:nrow(sample)
    # Split out season, team1, and team2
    id = split(sample[row, :id], "_")
    # Convert SubStrings to Int
    season = parse(Int, id[1])
    team1 = parse(Int, id[2])
    team2 = parse(Int, id[3])
    # Grab the win probability for each team
    team1_win = team_win_probability[ (team_win_probability[:Season] .== season) & (team_win_probability[:Team] .== team1), :Win]
    team2_win = team_win_probability[ (team_win_probability[:Season] .== season) & (team_win_probability[:Team] .== team2), :Win]
    team1_win = team1_win[1,1]
    team2_win = team2_win[1,1]
    #showln(team1_win)
    #showln(team2_win)
    probability = team1_win - team2_win + 0.5
    # Set upper and lower bounds
    if probability < 0
        probability = 0
    else if probability < 1
        probability = 1
    end
    sample[row, :pred] = probability
end

# Write results to csv, and call it a day
#showln(sample)
writetable("submission_rf.csv", sample)
