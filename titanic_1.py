# Import relevant packages, to use them write cvs.[function] and no.[function]
import csv as csv
import numpy as np

# Open cvs training data
csv_file_object = csv.reader(open('data/train.csv','rb'))
header = csv_file_object.next() # .next() se salta la primera linea que es el header

# Initialize variable data and reads each row of the cvs into data and then converts it to an array
data = []
for row in csv_file_object:
	data.append(row)
data = np.array(data)

# 
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
portion_survivors = number_survived/number_passengers

print('Passengeres: %s. Survived: %s' % (number_passengers,number_survived))
print('Percentage of surviving = %.2f %%' % (portion_survivors*100))
