# My attempt to model the March Madness brackets based on the past 4 seasons, and apply it to predicting the 2017 tournament

module BracketPredictor

using DataFrames

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

# Things to consider:
### 1) Do regular season statistics correlations predict victories?
##### 1a) One caveat is we do not know quality of play (major vs mid-major conference)
### 2) Do these translate to similar predictions in the tournament?
### 3) Establish an ELO ranking system potentially
##### 3a) Use end of season tournament seeds (seed #s 1-10) to assign early ELO to next season.
#####     Start at 1600 for 1-seed, decrement by 10 per seed increase, everyone else is 1500
##### 3b) Get stats-per-game for each team in that season, and use correlation and ELO to assign probability

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

nrows, ncols = size(team_stats_by_season)
for row in 1:nrows
  for col in 3:ncols
    team_stats_by_season[row,col] = team_stats_by_season[row,col] / total_team_games[row,:TotalGames]
  end
end
names!(team_stats_by_season, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF])
showln(team_stats_by_season)


end # module BracketPredictor
