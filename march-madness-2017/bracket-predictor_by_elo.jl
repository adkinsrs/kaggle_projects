# This script will take the submission data from bracket-predictor-by-stats.jl and will use ELO to influence the win probabilities even more

# I would love to track daily ELO change, but for the purposes of this, I want to keep track of ELO up until tournament time.

using DataFrames

# So I have a proposal to jump-start ELO variance
# ELO at the start of a season will be influenced by seeding in the tournament
# Only the beat 10 seeds (40 teams) will get the boost
# Typically worse seeds are automatic bid qualifiers
function increase_ELO_for_seeds()

end

# Stole function from
# https://www.kaggle.com/kplauritzen/march-machine-learning-mania-2017/elo-ratings-in-python
function update_elo(winner_elo, loser_elo):
    # https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details

    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo
end

# Stole function from
# https://www.kaggle.com/kplauritzen/march-machine-learning-mania-2017/elo-ratings-in-python
function expected_result(elo_a, elo_b):
    # https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details

    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a
end

### MAIN ###
function main()
    if length(ARGS) != 1
        println("Need to provide submisison file as argument")
        exit(1)
    end
    submission = ARGS[1]

    season_results = readtable("./input/RegularSeasonCompactResults.csv")
    tourney_results = readtable("./input/TourneyCompactResults.csv")
    seeds = readtable("./input/TourneySeeds.csv")
    teams = readtable("./input/Teams.csv")

    # Drop unneeded columns
    delete!(season_results, [:Wscore, :Lscore, :Wloc, :Numot])
    delete!(tourney_results, [:Wscore, :Lscore, :Wloc, :Numot])

    # Only consider data from 2010 onwards.  I want to give enough of a window for the mid-major conferences to lower their ELO compared to a major conference
    season_results = season_results[ season_results[:Season] >= 2010],:] ]
    tourney_results = tourney_results[ tourney_results[:Season] >= 2010],:] ]
    tourney_seeds = tourney_seeds[ tourney_seeds[:Season] >= 2010],:] ]

    # To start initial ELO adjustments, need tourney seeding from 2009
    initial_seeds = tourney_seeds[ tourney_seeds[:Season] == 2009],:] ]

    # Estabilish initial ELO for teams
    teams[:ELO] = fill(1500, nrow(teams))

    # Before the season's tournament, write end-of-season ELO for each tourney team
    tourney_seeds[:ELO] = fill(1500, nrow(tourney_seeds))

    # To get some initial variance, increase ELO for top 40 teams in 2009 tournament
    teams = increase_ELO_for_seeds()

end

main()
quit()
