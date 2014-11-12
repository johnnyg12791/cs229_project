#Build training data

#Our first attempt at real life ML...
import numpy as np
import matplotlib
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


global team_name_to_id
global games_matrix

def main():
    BuildTeamNameToIdMap('data/kaggle/teams.csv')
    #1 = id, 18 = FG%, 24 = FT%
    basic_stats_matrix = ReadBasicStatsToMatrix('data/sports-reference/basic_stats.csv', [1, 18, 24])
    BuildGamesMatrix('data/kaggle/regular_season_results.csv', 75290, 80543)
    #print games_matrix

    #So now I have the 3 different data sources into numpy matricies
    #combine the games matrix and stats matrix
    training_data = buildTrainingDataMatrix(basic_stats_matrix)
    (X,y) = (training_data[:,[0,1,2,3]], training_data[:,4])
    print X
    print y

    model = LogisticRegression()
    model = model.fit(X, y)

    print model.predict_proba([.23, .33, .78, .83]) #look, we think team 1 will lose
    print model.predict_proba([.55, .76, .34, .52]) #we think team 1 will win





'''
I want to create one matrix that uses all three of those to build a matrix
[team1_fg%, team2_fg%, team1_fg%, team2_fg%, team1_win?(0/1)]
'''
def buildTrainingDataMatrix(stats_matrix):
    training_data_matrix = np.zeros(shape = (len(games_matrix), 5))
    for game_index, game in enumerate(games_matrix):
        team1 = game[0]
        team2 = game[1]
        win_loss = game[2]

        for stats_index, team_id in enumerate(stats_matrix[:,0]):
            if(team_id == team1):
                #print team_id
                #print game
                #print stats_matrix[stats_index]
                #raw_input("")
                training_data_matrix[game_index][0] = stats_matrix[stats_index][1]
                training_data_matrix[game_index][1] = stats_matrix[stats_index][2]
            if(team_id == team2):
                training_data_matrix[game_index][2] = stats_matrix[stats_index][1]
                training_data_matrix[game_index][3] = stats_matrix[stats_index][2]
        training_data_matrix[game_index][4] = win_loss
    
    #Remove rows that didn't correspond to a team (naming issues between datasets)
    rows_to_delete = []
    for row_index, data_row in enumerate(training_data_matrix):
        if(data_row[0] == 0 or data_row[2] == 0):
            rows_to_delete.append(row_index)

    #now return the training data with non-filled rows deleted
    return np.delete(training_data_matrix, rows_to_delete, 0) #delete the rows going backwards


'''
Takes the regular season games (from kaggle), and builds a numpy matrix of the form:
[team1_id, team2_id, team1_win (0/1), season, daynum]
'''
def BuildGamesMatrix(filename, start_index, end_index):
    content = open(filename).read().splitlines()[start_index:end_index]
    global games_matrix
    games_matrix = np.zeros(shape = (len(content), 5)) #(rows, cols)

    for row_index, row in enumerate(content):
        line = row.split(",")

        #swap every other game as a win/loss
        if(row_index % 2 == 0):
            games_matrix[row_index][0] = line[2]
            games_matrix[row_index][1] = line[4]
            games_matrix[row_index][2] = 1
        else:
            games_matrix[row_index][0] = line[4]
            games_matrix[row_index][1] = line[2]
            games_matrix[row_index][2] = 0
        #season, daynum
        games_matrix[row_index][3] = ord(line[0]) #converts the 'season' letter to a number
        games_matrix[row_index][4] = line[1]




'''
Takes the given file (kaggle teams), and puts it into a map from name to id
'''
def BuildTeamNameToIdMap(filename):
    global team_name_to_id
    team_name_to_id = {}
    content = open(filename).read().splitlines()[1:]
    for row in content:
        row_split = row.split(",")
        team_name_to_id[row_split[1]] = int(row_split[0])



'''
Takes the 'basic stats' file and turns it into a matrix with the passed in columns
Indexing the team name based on our kaggle data
matrix looks like:
[team_id, stat1, stat2....]
'''
def ReadBasicStatsToMatrix(filename, columns):
    #print team_name_to_id
    content = open(filename).read().splitlines()[2:]
    matrix = np.zeros(shape = (len(content), len(columns))) #(rows, cols)
    for row_index, row in enumerate(content):
        row_as_list = row.split(",")
        #print row_as_list
        for col_index, col in enumerate(columns):
            if(col == 1):
                team_name = row_as_list[col]
                if(team_name in team_name_to_id):
                    matrix[row_index][col_index] = team_name_to_id[team_name]
                else:
                    matrix[row_index][col_index] = -1 #this is what we put if the name doesn't match for now
            else:
                matrix[row_index][col_index] = float(row_as_list[col])

    #might want to remove all the teams with a -1 index, but leave that for later if necessary...?
    return matrix



if __name__ == "__main__":
    main()