import csv
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from sklearn import preprocessing

dataset = []
labels = ['year', 'platform', 'genre', 'mode', 'group', 'type']
numbers = {"platform": ['pc', 'console', 'mobile'],
           "genre": ['action','adventure','rpg','simulation','strategy','puzzle','sports','platformer','shooter','racing','roguelike','running'],
           "mode": ['single', 'multi', 'online'],
           "group": ['production', 'management-people', 'management-feature', 'business', 'monetization'],
           "type": ['bugs', 'design', 'documentation', 'prototyping', 'techincal', 'testing','tools','communication','crunch-time', 'delays', 'team', 'cutting-features','feature-creep', 'multiple-projects', 'budget', 'planning', 'security', 'scope', 'marketing', 'monetization']
           }
           
def convertToNumber(typ, value):
    return int(numbers[typ].index(value))

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

with open('dataset.csv', encoding = 'utf8') as f:
    reader = csv.reader(f)
    header_row = next(reader)
    print(header_row)
    for row in reader:
        try:
            tempTable = [int(row[2]),
                         convertToNumber('platform', row[5]),
                         convertToNumber('genre', row[6]),
                         convertToNumber('mode', row[7]),
                         convertToNumber('group', row[8]),
                         convertToNumber('type', row[9])]
            dataset.append(tempTable)
            #print(tempTable)
        except ValueError:
            error = row[3]
        else:
            error = row[3]

    normalized = preprocessing.normalize(dataset, axis = 0)
    prediction = predict_classification(dataset, dataset[0], 3)
    #print(normalized)
    print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
