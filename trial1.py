#Our first attempt at real life ML...
import csv


def ReadFileInAsMatrix(file1):
    '''with open(file1, 'rb') as csvfile:
        filereader = csv.reader(csvfile)
        for row in filereader:
            print row
            #print ' '.join(row)
            '''
    content = open(file1).read().splitlines()
    for row in content:
        print row.split(",")

def main():
    print "test"
    ReadFileInAsMatrix('data/sports-reference/2012_margin_of_victory.csv')

        #'data/kaggle/teams.csv')

    # data/sports-reference/2012_margin_of_victory.csv

    #[team, avg_margin]
    #[team2, avg_margin..]...345]
  
if __name__ == "__main__":
    main()
