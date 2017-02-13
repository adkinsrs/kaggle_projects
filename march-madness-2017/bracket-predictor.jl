# My attempt to model the March Madness brackets based on the past 4 seasons, and apply it to predicting the 2017 tournament

module BracketPredictor

using DataFrames

showln(x) = (show(x); println())    # Quick function to show variable on its own line

# Load in the team IDs
teams = readtable("./input/teams.csv")
season_stats = readtable("./input/RegularSeasonDetailedResults.csv")
tourney_stats = readtable("./input/TourneyDetailedResults.csv")
seeds = readtable("./input/TourneySeeds.csv")
slotw = readtable("./input/TourneySlots.csv")

# Things to consider:
### 1) Do regular season statistics correlations predict victories?
##### 1a) One caveat is we do not know quality of play (major vs mid-major conference)
### 2) Do these translate to similar predictions in the tournament?
### 3) Establish an ELO ranking system potentially
##### 3a) Use end of season tournament seeds (seed #s 1-10) to assign early ELO to next season.  
#####     Start at 1600 for 1-seed, decrement by 10 per seed increase, everyone else is 1500
##### 3b) Get stats-per-game for each team in that season, and use correlation and ELO to assign probability

println("Columns in the seasons_stats (detailed) dataset:")
showcols(season_stats)
println(@sprintf("Number of rows: %d", nrow(season_stats)))

