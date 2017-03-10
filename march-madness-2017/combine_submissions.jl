#!/usr/bin/env Julia

# This script will combine the previous submission models, and average their probabilities for a conclusive probability.

using DataFrames

combined = DataFrame()

# Gather all submission tables, and join on match ID
for i in 1:length(ARGS)
    submission = readtable(ARGS[i])
    if i > 1
        combined = join(combined, submission, on= :id)
    else
        combined = submission
    end
end

# Just clone one of the tables for now
final_submission = readtable(ARGS[1])
# Get mean for range of prediction columns
final_submission[:pred] = mean( [combined[i] for i in 2:ncol(combined)] )

# Write results to csv, and call it a day
writetable("submission_final.csv", final_submission)
quit()
