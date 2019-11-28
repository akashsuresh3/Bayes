import csv
import random
import math
import numpy as np
from sklearn.metrics import confusion_matrix

def fileload(filename):
	setofpoints=csv.reader(open(filename,'r'))
	next(setofpoints)
	dataset=list(setofpoints)
	for l in range(len(dataset)):
		dataset[l]=[float(dataset[l][x]) for x in range(1,len(dataset[l]))]
	return dataset


def gettrain(dataset, splitratio):
	trainSize=int(len(dataset)*splitratio)
	trainingset=[]	
	copy=list(dataset)
	while len(trainingset) < trainSize:
		index=random.randint(0,len(copy)-1)
		trainingset.append(copy.pop(index))
	return [trainingset, copy]


def calc(dataset):
	updatedclass=[]
	for attribute in zip(*dataset):
		x=sum(attribute)/float(len(attribute))		                        # x is the mean
		z=sum([pow(a-x,2) for a in attribute])/float(len(attribute))		# z is the variance
		y=math.pow(z,0.5)		                        		# y is the standard deviation
		if (y==0):
			y=0.001
		updatedclass.append([x,y])
	del updatedclass[-1]
	return updatedclass

def dividetotwo(dataset):
	dividedclass={}
	dividedclass[0]=[]	
	dividedclass[1]=[]
	for i in range(len(dataset)):
		vector=dataset[i]
		if (vector[-1] == 0):
			dividedclass[0].append(vector)			
		elif(vector[-1]==1):
			dividedclass[1].append(vector)
	return dividedclass	

def groupprob(dataset):
	divided=dividetotwo(dataset)
	probval={}
	for classValue, instances in divided.items():
		probval[classValue]=calc(instances)
	return probval

def normaldistpdf(x,mean,stdev):
	epowr=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1/(math.sqrt(2*math.pi)*stdev))*epowr

def eachclassprob(ProcessValues, inputVector):
	probabilities={}
	for classValue, classSummaries in ProcessValues.items():
		probabilities[classValue]=1
		for i in range(len(classSummaries)):
			mean,stdev=classSummaries[i] 
			x=inputVector[i]
			probabilities[classValue]*=normaldistpdf(x,mean,stdev)	
	return probabilities

def predictprob(ProcessValues,inputVector):
	probabilities=eachclassprob(ProcessValues,inputVector)
	bestLabel, bestProb=None, -1
	for classValue,probability in probabilities.items():
		if bestLabel is None or probability>bestProb:
			bestProb=probability
			bestLabel=classValue
	return bestLabel

def storepred(ProcessValues,testSet):
	predictions=[]
	y_true=[]	
	for i in range(len(testSet)):
		result=predictprob(ProcessValues,testSet[i])
		predictions.append(result)

	for i in range(len(testSet)):
		vector=testSet[i]
		y_true.append(vector[-1])
	return [y_true,predictions]

def clacacuracy(testSet,predictions):
	exact=0
	for i in range(len(testSet)):
		if testSet[i][-1]==predictions[i]:
			exact+=1
	return (exact/float(len(testSet)))*100.0

def main():
	file='catalog1/cat1.csv'
	'''sRatio1=0.80
	   sRatio2=0.60
	   sRatio3=0.40'''
	splitratio=[0.80,0.60,0.40]
	dataset=fileload(file)
	for i in range(3):
		print('\nFOR SRATIO = ',splitratio[i])
		training,test=gettrain(dataset,splitratio[i])
		PV=groupprob(training)
		y_true, predictions=storepred(PV,test)
		cm=confusion_matrix(y_true, predictions)
		print('\n\n Confusion Matrix \n')
		print('\n'.join([''.join(['{:8}'.format(item) for item in row]) for row in cm]))
		FP=cm.sum(axis=0) - np.diag(cm)
		FN=cm.sum(axis=1) - np.diag(cm)
		TP=np.diag(cm)
		TN=cm.sum()-(FP+FN+TP)
		print('False Positives\n {}'.format(FP))
		print('False Negatives\n {}'.format(FN))
		print('True Positives\n {}'.format(TP))
		print('True Negatives\n {}'.format(TN))
		TPR=TP/(TP+FN)
		print('Sensitivity \n {}'.format(TPR))
		TNR=TN/(TN+FP)
		print('Specificity \n {}'.format(TNR))
		Precision=TP/(TP+FP)
		print('Precision \n {}'.format(Precision))
		Recall=TP/(TP+FN)
		print('Recall \n {}'.format(Recall))
		Acc=(TP+TN)/(TP+TN+FP+FN)
		print('Accuracy\n',Acc[0])
		Fscore=2*(Precision*Recall)/(Precision+Recall)
		print('Fscore \n{}'.format(Fscore))
		print(len(dataset))
		

main()



