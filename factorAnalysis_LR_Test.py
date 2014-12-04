#Build training data
#Using margins of victory and RPI (hopefully)

#Our second attempt at real life ML...
import numpy as np
import matplotlib, itertools
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


global team_name_to_id
global factors_to_results
#global games_matrix

def main():
    BuildTeamNameToIdMap('data/kaggle/teams.csv')
    #1 = id, 5 = W/L %, 6 = SRS, 18 = FG%, 21 = 3PFG%, 24 = FT%, 32 = Avg Margin
    #33 = TRB/G, 34 = AST/G, 35 = TOV/G, 36 = FLS/G, 37 = STl/G, 38 = Blk/G
    ###basic_stats_matrix = ReadBasicStatsToMatrix('data/sports-reference/basic_stats.csv', [1, 5, 6, 18, 21, 24, 32, 33, 34, 35, 36, 37, 38]) #
    ###regular_season_games = BuildGamesMatrix('data/kaggle/regular_season_results.csv', 75290, 80543)
    #print regular_season_games
    #print games_matrix

    #So now I have the 3 different data sources into numpy matricies
    #combine the games matrix and stats matrix
    """training_data = buildTrainingDataMatrix(basic_stats_matrix, regular_season_games, 12)
    print training_data
    (X,y) = (training_data[:,range(24)], training_data[:,24]) #
    #print X
    #print y

    model = LogisticRegression()
    model = model.fit(X, y)

    model2 = svm.SVC()
    model2 = model2.fit(X,y)"""

    #print model.predict_proba([1.75, .12])
    #print model2.predict([1.75, .12]) #look, we think team 1 will lose
    #print model.predict_proba([.55, .76, .34, .52]) #we think team 1 will win

    ###tournament_games = BuildGamesMatrix('data/kaggle/tourney_results.csv', 1024, 1090)
    
    #test on tourny and test on regular season
    ###test_data = buildTrainingDataMatrix(basic_stats_matrix, tournament_games, 12)
    #test_data = buildTrainingDataMatrix(basic_stats_matrix, regular_season_games)
    #this is training_data!!!!

    #print len(test_data) only get 32 of the games due to naming....
    """predictions = MakePredictions(model, test_data[:,range(24)]) #
    accuracy = GetAccuracy(predictions, test_data, 12)
    print "LogisticRegression Accuracy: ", accuracy

    predictions2 = MakePredictionsSVM(model2, test_data[:,range(24)])
    accuracy2 = GetAccuracy(predictions2, test_data, 12)
    print "SVM Accuracy: ", accuracy2"""
    FactorAnalysis()


def FactorAnalysis():
    global factors_to_results
    factors_to_results = {}

    regular_season_games = BuildGamesMatrix('data/kaggle/regular_season_results.csv', 75290, 80543)
    tournament_games = BuildGamesMatrix('data/kaggle/tourney_results.csv', 1024, 1090)

    factors = [5, 6, 18, 21, 24, 32, 33, 34, 35, 36, 37, 38]
    i = 1

    while (i <= 12): 
        for combo in itertools.combinations(factors, i):
            combo = list(combo)
            combo.insert(0,1)
            
            MakeResultsDictionary(regular_season_games, tournament_games, combo, i)
            #print combo
        i += 1

    #PrintTop5Results()

def MakeResultsDictionary(regular_season_games, tournament_games, factorCombo, numFac):
    basic_stats_matrix = ReadBasicStatsToMatrix('data/sports-reference/basic_stats.csv', factorCombo) #

    training_data = buildTrainingDataMatrix(basic_stats_matrix, regular_season_games, numFac)
    test_data = buildTrainingDataMatrix(basic_stats_matrix, tournament_games, numFac)
    (X,y) = (training_data[:,range(2*numFac)], training_data[:,2*numFac]) 

    model = LogisticRegression()
    model = model.fit(X, y)

    #model2 = svm.SVC()
    #model2 = model2.fit(X,y)

    # Make predictions and get accuracies 
    # LogisticTest, SVMTest, LogisticTrain, SVMTrain
    prediction1 = MakePredictions(model, test_data[:,range(2*numFac)]) 
    accuracy1 = GetAccuracy(prediction1, test_data, numFac)
    
    '''prediction2 = MakePredictionsSVM(model2, test_data[:,range(2*numFac)])
    accuracy2 = GetAccuracy(prediction2, test_data, numFac)

    prediction3 = MakePredictions(model, training_data[:,range(2*numFac)]) 
    accuracy3 = GetAccuracy(prediction3, training_data, numFac)
    
    prediction4 = MakePredictionsSVM(model2, training_data[:,range(2*numFac)])
    accuracy4 = GetAccuracy(prediction4, training_data, numFac)'''

    factors_to_results[tuple(factorCombo)] = accuracy1
    print factors_to_results[tuple(factorCombo)]


def PrintTop5Results():
    for factorCombo in factors_to_results:
        print factors_to_results[factorCombo]


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
                #print team_id
                #print game
                #print stats_matrix[stats_index]
                #raw_input("")
                i = 0
                while (i < numFac):
                    training_data_matrix[game_index][i] = stats_matrix[stats_index][i+1]
                    i += 1

                """training_data_matrix[game_index][0] = stats_matrix[stats_index][1] 
                training_data_matrix[game_index][1] = stats_matrix[stats_index][2] #
                training_data_matrix[game_index][2] = stats_matrix[stats_index][3] 
                training_data_matrix[game_index][3] = stats_matrix[stats_index][4]
                training_data_matrix[game_index][4] = stats_matrix[stats_index][5]
                training_data_matrix[game_index][5] = stats_matrix[stats_index][6] 
                training_data_matrix[game_index][6] = stats_matrix[stats_index][7] #
                training_data_matrix[game_index][7] = stats_matrix[stats_index][8] 
                training_data_matrix[game_index][8] = stats_matrix[stats_index][9]
                training_data_matrix[game_index][9] = stats_matrix[stats_index][10] #
                training_data_matrix[game_index][10] = stats_matrix[stats_index][11] 
                training_data_matrix[game_index][11] = stats_matrix[stats_index][12]"""  
            if(team_id == team2):
                j = 0
                while (j < numFac):
                    training_data_matrix[game_index][j+numFac] = stats_matrix[stats_index][j+1]
                    j += 1

                """training_data_matrix[game_index][12] = stats_matrix[stats_index][1] #
                training_data_matrix[game_index][13] = stats_matrix[stats_index][2] #
                training_data_matrix[game_index][14] = stats_matrix[stats_index][3] 
                training_data_matrix[game_index][15] = stats_matrix[stats_index][4]
                training_data_matrix[game_index][16] = stats_matrix[stats_index][5]
                training_data_matrix[game_index][17] = stats_matrix[stats_index][6] 
                training_data_matrix[game_index][18] = stats_matrix[stats_index][7] #
                training_data_matrix[game_index][19] = stats_matrix[stats_index][8] 
                training_data_matrix[game_index][20] = stats_matrix[stats_index][9]
                training_data_matrix[game_index][21] = stats_matrix[stats_index][10] #
                training_data_matrix[game_index][22] = stats_matrix[stats_index][11] 
                training_data_matrix[game_index][23] = stats_matrix[stats_index][12]"""  
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