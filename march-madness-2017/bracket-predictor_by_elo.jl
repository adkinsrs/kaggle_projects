#!/usr/bin/env Julia

# This script will take the submission data from bracket-predictor-by-stats.jl and will use ELO to influence the win probabilities even more

# I would love to track daily ELO change, but for the purposes of this, I want to keep track of ELO up until tournament time.

using DataFrames

### Constants ###
mean_elo = 1500
elo_width = 400
k_factor = 64

showln(x) = (show(x); println())    # Quick function to show variable on its own line

# So I have a proposal to jump-start ELO variance
# ELO at the start of a season will be influenced by seeding in the tournament
# Only the beat 10 seeds (40 teams) will get the boost
# Typically worse seeds are automatic bid qualifiers
function increase_ELO_for_seeds(teams, initial_seeds)
    top_elo = 1600

    for r in 1:nrow(initial_seeds)
        seed = initial_seeds[r, :Seed]
        seed = match(r"\d+", seed) # Get the seed digit
        seed = parse(Int, seed.match)

        team = initial_seeds[r, :Team]
        # Top seed gets ELO of 1600 to start.  Worse seeds get lower ELO scores in 10-pt decrements, down to 1510 for a 10-seed
        if seed <= 10
            teams[ teams[:Team_Id] .== team, :ELO ] = top_elo - ((seed-1) * 10)
        end
    end
#showln(teams[ teams[:ELO] > 1500, :])
    return teams
end

# Stole function from
# https://www.kaggle.com/kplauritzen/march-machine-learning-mania-2017/elo-ratings-in-python
function update_elo(winner_elo, loser_elo)
    # https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    #ELOs are now in Float, but I'd rather just keep them in Int
    return winner_elo, loser_elo
end

# Stole function from
# https://www.kaggle.com/kplauritzen/march-machine-learning-mania-2017/elo-ratings-in-python
function expected_result(elo_a, elo_b)
    # https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details

    # Using 'div' instead of '/' to force Julia to return an Int instead of Float
    expect_a = div(1, (1 + 10.0^ div(elo_b - elo_a, elo_width) ) )
    return expect_a
end

# Stole function from
# https://www.kaggle.com/kplauritzen/march-machine-learning-mania-2017/elo-ratings-in-python
function update_end_of_season(teams)
    #https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/

    # Using 'div' instead of '/' to force Julia to return an Int instead of Float
    teams[:ELO] = map(x -> x - div(x - mean_elo ,3), teams[:ELO])
    return teams
end

# Stole much of the code from
# https://www.kaggle.com/kplauritzen/march-machine-learning-mania-2017/elo-ratings-in-python
function simulate_games(games, teams)
    for row in 1:nrow(games)
        w_id = games[row, :Wteam]
        l_id = games[row, :Lteam]
        w_elo_before = teams[ teams[:Team_Id] .== w_id, :ELO ]
        l_elo_before = teams[ teams[:Team_Id] .== l_id, :ELO ]
        w_elo_after, l_elo_after = update_elo(w_elo_before[1,1], l_elo_before[1,1])

        teams[ teams[:Team_Id] .== w_id, :ELO] = w_elo_after
        teams[ teams[:Team_Id] .== l_id, :ELO] = l_elo_after
    end
    return teams
end

# Record ELO between regular season and tournament
function save_tourney_seeds_ELO(tourney_seeds, teams, year)
    # wanted to do a one-liner or a map function but oh well
    for i in 1:nrow(tourney_seeds)
        if tourney_seeds[ i, :Season] == year
            team = tourney_seeds[i, :Team]
            team_elo = teams[ teams[:Team_Id] .== team, :ELO ]
            tourney_seeds[i, :ELO] = team_elo[1,1]
        end
    end
    return tourney_seeds
end

# Calculate the probabilites to write in the current submission file
function calc_submission_probs(tourney_seeds, teams, submission)
    for row in 1:nrow(submission)
        # Split out season, team1, and team2
        id = split(submission[row, :id], "_")
        # Convert SubStrings to Int
        season = parse(Int, id[1])
        team1 = parse(Int, id[2])
        team2 = parse(Int, id[3])

        # Grab the ELOs for each team
        team1_elo = tourney_seeds[ (tourney_seeds[:Season] .== season) & (tourney_seeds[:Team] .== team1), :ELO][1,1]
        team2_elo = tourney_seeds[ (tourney_seeds[:Season] .== season) & (tourney_seeds[:Team] .== team2), :ELO][1,1]

        probability = team1_elo / (team1_elo + team2_elo)
        submission[row, :pred] = probability
    end
    return submission
end

### MAIN ###
function main()
    if length(ARGS) != 1
        println("Need to provide submisison file as argument")
        exit(1)
    end
    submission = readtable(ARGS[1])

    season_results = readtable("./input/RegularSeasonCompactResults.csv")
    tourney_results = readtable("./input/TourneyCompactResults.csv")
    seeds = readtable("./input/TourneySeeds.csv")
    teams = readtable("./input/Teams.csv")

    # Drop unneeded columns
    delete!(season_results, [:Wscore, :Lscore, :Wloc, :Numot])
    delete!(tourney_results, [:Wscore, :Lscore, :Wloc, :Numot])

    # Only consider data from 2010 onwards.  I want to give enough of a window for the mid-major conferences to lower their ELO compared to a major conference
    season_results = season_results[ season_results[:Season] .>= 2010,: ]
    tourney_results = tourney_results[ tourney_results[:Season] .>= 2010,: ]
    tourney_seeds = seeds[ seeds[:Season] .>= 2010,: ]

    # To start initial ELO adjustments, need tourney seeding from 2009
    initial_seeds = seeds[ seeds[:Season] .== 2009,: ]

    # Estabilish initial ELO for teams
    teams[:ELO] = fill(mean_elo, nrow(teams))

    # Before the season's tournament, write end-of-season ELO for each tourney team
    tourney_seeds[:ELO] = fill(mean_elo, nrow(tourney_seeds))

    # To get some initial variance, increase ELO for top 40 teams in 2009 tournament
    teams = increase_ELO_for_seeds(teams, initial_seeds)

    for i in 2010:2016
        # Iterate through one season
        one_season = season_results[ season_results[:Season] .== i,: ]
        teams = simulate_games(one_season, teams)

        tourney_seeds = save_tourney_seeds_ELO(tourney_seeds, teams, i)

        # Iterate through one tournament
        one_tourney = tourney_results[ tourney_results[:Season] .== i,: ]
        teams = simulate_games(one_tourney, teams)

        # Update ELO scores in between seasons to regress towards the mean
        teams = update_end_of_season(teams)
    end

    submission = calc_submission_probs(tourney_seeds, teams, submission)
    # Write results to csv, and call it a day
    writetable("submission_using_elo.csv", submission)
end

main()
quit()
