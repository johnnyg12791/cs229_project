#Baseline data

#Go through the tournament results, tournament seeds data from kaggle and come up with a baseline for how often higher seeds win
#input in a tournament year and see the % of higher seeds that win
#optional argument of teams not to count???

import sys
import numpy as np



global year
global seed_difference

def main():
    higher_seed_win_percentage = {}
    tourney_seeds = ReadTournamentSeedData("data/kaggle/tourney_seeds.csv")
    tourney_results = ReadTournamentResults("data/kaggle/tourney_results.csv")
    #print tourney_seeds
    #print tourney_results
    global seed_difference
    seed_difference = 0
    #year = 2013
    #year = year - 1985 #because 1985 is represented as year 0 in our data
    year_to_win_percentage = BuildBaselineWinPercentageDict(tourney_seeds, tourney_results)
    #for season in range(1985, 2014):
        #print season
    #    pass



'''
create dictionary, season -> %% higher seed wins
'''

def BuildBaselineWinPercentageDict(seeds, results):
    accuracy = {}
    for season in range(0, 18):
        num_higher_seed_wins = 0.0
        total_different_seeded_games = 0.0
        season_results = results[results[:,0] == season]
        #print season_results
        #raw_input("")
        for result, win_team_id, lose_team_id in season_results:
            winning_team_seed = seeds.get((season, win_team_id))
            losing_team_seed = seeds.get((season, lose_team_id))
            if(winning_team_seed < losing_team_seed - seed_difference):
                num_higher_seed_wins += 1.0
                total_different_seeded_games += 1.0
            elif(winning_team_seed > losing_team_seed + seed_difference):
                total_different_seeded_games += 1.0
        accuracy[season+1996] = num_higher_seed_wins/total_different_seeded_games
        #raw_input("")
    print accuracy


'''
This method will take the tournament seed file and read it into a dictionary
[season, team_id] -> seed
'''
def ReadTournamentSeedData(filename):
    content = open(filename).read().splitlines()[1:]
    #matrix = np.zeros(shape = (len(content), len(content[0].split(",")))) #(rows, cols)
    seed_data = {}
    for row_index, row in enumerate(content):
        row_as_list = row.split(",")
        season = ord(row_as_list[0]) - ord('A') #This converts the season to a number
        seed = int(row_as_list[1][1:3]) #This converts the seed to an int, removing region (and play in game)
        team_id = int(row_as_list[2]) #This is the team ID
        seed_data[(season, team_id)] = seed
    return seed_data


'''
This method takes the tournament results file and converts it into a numpy matrix
season, winning team, losing team
'''
def ReadTournamentResults(filename):
    content = open(filename).read().splitlines()[1:]
    matrix = np.zeros(shape = (len(content), 3)) #(rows, cols)
    for row_index, row in enumerate(content):
        row_as_list = row.split(",")
        matrix[row_index][0] = ord(row_as_list[0]) - ord('A') #This converts the season to a number
        matrix[row_index][1] = int(row_as_list[2]) #This is the winning team
        matrix[row_index][2] = int(row_as_list[4]) #This is the losing team
    return matrix


if __name__ == "__main__":
    main()