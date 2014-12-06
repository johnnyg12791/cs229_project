#Build training data
#Using margins of victory and RPI (hopefully)

#Our second attempt at real life ML...
import sys
import numpy as np
import matplotlib, itertools
import operator
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


global team_name_to_id
global factors_to_results
global model_flag


def main():
    global model_flag
    model_flag = sys.argv[1]
    global year_flag
    year_flag = int(sys.argv[2])

    BuildTeamNameToIdMap('data/kaggle/teams.csv')
    #FactorAnalysis()
    GetResults()
    #WriteToOutputFile()

def GetResults():
    regular_season_games = BuildGamesMatrixWithYear('data/kaggle/regular_season_results.csv')
    tournament_games = BuildGamesMatrixWithYear('data/kaggle/tourney_results.csv')

    factors = [1, 5, 6, 14, 15, 18, 21, 24]
    #There is a strange phenomena, when the CSV is copied from sports-ref/cbb
    #It inserts an empty column (2 commas with nothing between them) after col 15
    #So we need to increase those factors by 1. This occurs in basic stats (except 2012)
    factors = [1, 5, 6, 14, 15, 19, 22, 25]
    test_acc, train_acc = GetTestAndTrainAccuracy(regular_season_games, tournament_games, factors)
    print "Test Accuracy: ", test_acc
    print "Train Accuracy: ", train_acc

#Add a flag for how many to write? which ones
def WriteToOutputFile(index):
    f_train = open("features/" + str(year_flag) + "_" + model_flag + "_features_train_" + str(index) + ".out" ,'w')
    f_test = open("features/" + str(year_flag) + "_" + model_flag + "_features_test_" + str(index) + ".out", 'w')

    train_sorted_features = sorted(factors_to_results.keys(), key=lambda k: factors_to_results[k][0], reverse=True)
    test_sorted_features = sorted(factors_to_results.keys(), key=lambda k: factors_to_results[k][1], reverse=True)
    #train_sorted_features = sorted(factors_to_results.items(), key=operator.itemgetter(1,0), reverse=True)
    #test_sorted_features = sorted(factors_to_results.items(), key=operator.itemgetter(1,1), reverse=True)
    
    for key in train_sorted_features:
        f_train.write("-".join(map(str, list(key))))
        f_train.write(",")
        f_train.write(str(factors_to_results[key][0]))
        f_train.write("\n")

    for key in test_sorted_features:
        f_test.write("-".join(map(str, list(key))))
        f_test.write(",")
        f_test.write(str(factors_to_results[key][1]))
        f_test.write("\n")


    f_train.close()
    f_test.close()


'''
Given the a matrix regular season and tournament games, as well as a list of factors
It builds the LR or SVM model, and predicts the tournament (test) and regular season(train) games
'''
def GetTestAndTrainAccuracy(regular_season_games, tournament_games, factors):
    numFac = len(factors)-1

    basic_stats_matrix = ReadBasicStatsToMatrix("data/sports-reference/" + str(year_flag) +"_basic_stats.csv", factors) #
    training_data = buildTrainingDataMatrix(basic_stats_matrix, regular_season_games, numFac)
    test_data = buildTrainingDataMatrix(basic_stats_matrix, tournament_games, numFac)
    (X,y) = (training_data[:,range(2*numFac)], training_data[:,2*numFac]) 

    if(model_flag == "lr"):
        model = LogisticRegression()
        model = model.fit(X, y)

        test_prediction = MakePredictions(model, test_data[:,range(2*numFac)]) 
        test_accuracy = GetAccuracy(test_prediction, test_data, numFac)
    
        train_prediction = MakePredictions(model, training_data[:,range(2*numFac)]) 
        train_accuracy = GetAccuracy(train_prediction, training_data, numFac)

    if(model_flag == "svm"):
        model = svm.SVC()
        model = model.fit(X,y)

        test_prediction = MakePredictionsSVM(model, test_data[:,range(2*numFac)])
        test_accuracy = GetAccuracy(test_prediction, test_data, numFac)
        
        train_prediction = MakePredictionsSVM(model, training_data[:,range(2*numFac)])
        train_accuracy = GetAccuracy(train_prediction, training_data, numFac)

    return (test_accuracy, train_accuracy)


#Given a set of predictions (binary classification) and data, computes accuracy
def GetAccuracy(predictions, test_data, numFac):
    correct_predictions = 0
    for index, prediction in enumerate(predictions):
        if(prediction == test_data[:,2*numFac][index]): #
            correct_predictions += 1.0
    return correct_predictions / float(len(predictions))

#Predictions for an SVM
def MakePredictionsSVM(model, test_data):
    predictions = []
    for data in test_data:
        predict = model.predict(data)
        if(predict[0] == 0):
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions

#Predictions using a classification
def MakePredictions(model, test_data):
    predictions = []
    for data in test_data:
        predict = model.predict_proba(data)
        if(predict[0][0] > .5):
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions




'''
I want to create one matrix that uses all three of those to build a matrix
[team1_fg%, team2_fg%, team1_fg%, team2_fg%, team1_win?(0/1)]

Converted to use one statistic!!!
'''
def buildTrainingDataMatrix(stats_matrix, games_matrix, numFac):
    training_data_matrix = np.zeros(shape = (len(games_matrix), 2*numFac+1)) #
    for game_index, game in enumerate(games_matrix):
        team1 = game[0]
        team2 = game[1]
        win_loss = game[2]

        for stats_index, team_id in enumerate(stats_matrix[:,0]):
            if(team_id == team1):
                i = 0
                while (i < numFac):
                    training_data_matrix[game_index][i] = stats_matrix[stats_index][i+1]
                    i += 1

            if(team_id == team2):
                j = 0
                while (j < numFac):
                    training_data_matrix[game_index][j+numFac] = stats_matrix[stats_index][j+1]
                    j += 1

        training_data_matrix[game_index][2*numFac] = win_loss #
    
    #Remove rows that didn't correspond to a team (naming issues between datasets)
    rows_to_delete = []
    for row_index, data_row in enumerate(training_data_matrix):
        if(data_row[0] == 0 or data_row[numFac] == 0): #
            rows_to_delete.append(row_index)

    #now return the training data with non-filled rows deleted
    return np.delete(training_data_matrix, rows_to_delete, 0) #delete the rows going backwards


'''
Takes the regular season games (from kaggle), and builds a numpy matrix of the form:
[team1_id, team2_id, team1_win (0/1), season, daynum]
'''
def BuildGamesMatrix(filename, start_index, end_index):
    content = open(filename).read().splitlines()[start_index:end_index]
    #global games_matrix
    games_matrix = np.zeros(shape = (len(content)*2, 5)) #(rows, cols)

    for row_index, row in enumerate(content):
        line = row.split(",")

        #Count every game twice, as a win for team 1 and a loss for team 2
        games_matrix[row_index*2][0] = line[2]
        games_matrix[row_index*2][1] = line[4]
        games_matrix[row_index*2][2] = 1
        games_matrix[row_index*2][3] = ord(line[0]) - ord('A') #converts the 'season' letter to a number
        games_matrix[row_index*2][4] = line[1]
        
        #Loss for team 2
        games_matrix[row_index*2+1][0] = line[4]
        games_matrix[row_index*2+1][1] = line[2]
        games_matrix[row_index*2+1][2] = 0
        games_matrix[row_index*2+1][3] = ord(line[0]) - ord('A') #converts the 'season' letter to a number
        games_matrix[row_index*2+1][4] = line[1]

    return games_matrix


#Can use the year flag
def BuildGamesMatrixWithYear(filename):
    content = open(filename).read().splitlines()
    start_index, end_index = GetStartEndIndexForYear(content)
    #print content[start_index:end_index] #You can see, this gets the data we want
    return BuildGamesMatrix(filename, start_index, end_index)


#Searches through the content file, finds start and end index of that season
#Tested by John, this gets the correct indicies
def GetStartEndIndexForYear(content):
    season_letter = chr(ord('A') + year_flag - 1996)
    next_season_letter = chr(ord('A') + year_flag - 1996 + 1) #next season
    start_index = 0
    end_index = 0
    updated_start_index = 0
    updated_end_index = 0
    for line_index, line in enumerate(content):
        if(line[0] == season_letter and not updated_start_index):
            start_index = line_index
            updated_start_index = 1
        if(line[0] == next_season_letter and not updated_end_index):
            end_index = line_index
            updated_end_index = 1

    return (start_index, end_index)




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
    #Opens the stats file, gets lines 2 and beyond
    content = open(filename).read().splitlines()[2:]
    matrix = np.zeros(shape = (len(content), len(columns))) #(rows, cols)
    for row_index, row in enumerate(content):
        row_as_list = row.split(",")
        for col_index, col in enumerate(columns):
            #This is for the id only
            if(col == 1): 
                team_name = row_as_list[col]
                if(team_name in team_name_to_id):
                    matrix[row_index][col_index] = team_name_to_id[team_name]
                else:
                    #print "Uh oh we are missing a team"
                    matrix[row_index][col_index] = -1 #this is what we put if the name doesn't match for now
            
            else: #These are the rest of the statistics
                matrix[row_index][col_index] = float(row_as_list[col])

    #might want to remove all the teams with a -1 index, but leave that for later if necessary...?
    return matrix



if __name__ == "__main__":
    main()