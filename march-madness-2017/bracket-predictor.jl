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
winning_stats = by(season_stats, [:Season, :Wteam],
    df -> DataFrame( nrow, colwise(mean, df[ 9:21 ]) ))

# Next get stats of the losing team
losing_stats = by(season_stats, [:Season, :Lteam],
    df -> DataFrame( nrow, colwise(mean, df[ 22:34 ]) ))

# Renaming columns to allow for vcat to join properly
names!(winning_stats, [:Season, :Team, :Count, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
  :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF])
names!(losing_stats, [:Season, :Team, :Count, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
    :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF])

# Each team's seasonal statistics should be here now.
team_stats_by_season = vcat(winning_stats, losing_stats)
showln(team_stats_by_season)

#TODO: When calculating this mean, put more weight on the larger of wins/losses
team_stats_by_season = by(team_stats_by_season, [:Season, :Team],
    df -> DataFrame( colwise(mean, df[ 4:16 ]) ))
names!((team_stats_by_season, [:Season, :Team, :FGM, :FGA, :FGM3, :FGA3, :FTM, :FTA,
        :OffR, :DefR, :Assist, :TO, :Steal, :Block, :PF])
end # module BracketPredictor
