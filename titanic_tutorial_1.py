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

# Calculate portion of survivors
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
portion_survivors = number_survived/number_passengers
print('Passengeres = %s. Survived = %s' % (number_passengers,number_survived))
print('Percentage of surviving = %.2f %%' % (portion_survivors*100))

# Get indexes for women and men
women_ind = data[0::,4] == "female"
men_ind = data[0::,4] != "female"

# Size of women and men
num_women = np.size(data[women_ind,1].astype(np.float))
num_men = np.size(data[men_ind,1].astype(np.float))
print("Number of men = %.0f" % (num_men))
print("Number of women = %.0f" % (num_women))

# Percentage of surviving amog women and men
portion_women_surv = np.sum(data[women_ind,1].astype(np.float)) / num_women
portion_men_surv = np.sum(data[men_ind,1].astype(np.float)) / num_men
print("women survived = %.2f %%" %(portion_women_surv*100))
print("men survived = %.2f %%" %(portion_men_surv*100))

# Leer test
csv_test_object = csv.reader(open('data/test.csv','rb'))
header = csv_test_object.next()

# Crear archivo de predicciones
prediction_file = open("genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

# Escribir archivo
prediction_file_object.writerow(["PassengerId", "Survived"])
for row in csv_test_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then  
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                          # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
#csv_test_object.close()
prediction_file.close()
