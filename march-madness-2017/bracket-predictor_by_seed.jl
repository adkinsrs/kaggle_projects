#!/usr/bin/env Julia

# This script will predict winners for tournament games based on pure seeding

# Inspiration comes from https://www.kaggle.com/wacaxx/march-machine-learning-mania-2017/seed-benchmark-in-r/code#L37

using DataFrames

showln(x) = (show(x); println())    # Quick function to show variable on its own line


function calc_submission_probs(seeds, submission)
    for row in 1:nrow(submission)
        # Split out season, team1, and team2
        id = split(submission[row, :id], "_")
        # Convert SubStrings to Int
        season = parse(Int, id[1])
        team1 = parse(Int, id[2])
        team2 = parse(Int, id[3])

        seed_team1 = get_team_seed(seeds, season, team1)
        seed_team2 = get_team_seed(seeds, season, team2)

        probability =  0.5 + (seed_team2[1,1] - seed_team1[1,1]) * 0.03
        submission[row, :pred] = probability
    end
    return submission
end

# Given a team ID and season, return the seed #
function get_team_seed(seeds, season, team)
    seed = seeds[ (seeds[:Season] .== season) & (seeds[:Team] .== team), :Seed][1,1]
    seed = match(r"\d+", seed) # Get the seed digit
    seed = parse(Int, seed.match)
    return seed
end

function main()
    if length(ARGS) != 1
        println("Need to provide submisison file as argument")
        exit(1)
    end
    submission = readtable(ARGS[1])

    seeds = readtable("./input/TourneySeeds.csv")
    seeds = seeds[ seeds[:Season] .>= 2013,: ]

    submission = calc_submission_probs(seeds, submission)
    # Write results to csv, and call it a day
    writetable("submission_using_seeds.csv", submission)
end

main()
quit()
