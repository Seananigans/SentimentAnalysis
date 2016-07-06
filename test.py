import sqlite3
import time
import pandas as pd
import re
    
conn = sqlite3.connect('knowledgeBase.db')
conn.text_factory= str
c = conn.cursor()

# query = c.execute("SELECT * FROM imdb_reviews")
df = pd.read_sql("SELECT * FROM imdb_reviews", conn)
print df.value.min()
print df.value.mean()
print df.value.median()
print df.value.max()

df["sentiment"] = 1.0
df.ix[df.value<6,"sentiment"] = -1.0
df.ix[df.sentiment>0,"review"].to_csv("positiveSentiment.txt", index=False)
df.ix[df.sentiment<0,"review"].to_csv("negativeSentiment.txt", index=False)


negative_words = []
positive_words = []
word_dict = {}

sql = "SELECT * FROM wordVals WHERE value=?"

def loadWordArrays():
	for row in c.execute("SELECT * FROM wordVals"):
		word_dict[row[0]] = row[1]
	
def testSentiment(filename, weights=[-1,1]):
	readFile = open(filename,'r').read()
	split_read = readFile.split('\n')
	total_examples = len(split_read)
	posExamplesFound = 0
	number_correct=0
	
	for each_example in split_read:
		each_example = re.sub("\W"," ", each_example).lower()
		sentCounter = 0
		
		for word in each_example.split():
			word_value = word_dict.get(word, 0)
			word_value = weights[0] if word_value<0 else weights[1]
			sentCounter += word_value
	
		if (sentCounter>0 and filename.startswith("pos")) or (sentCounter<0 and filename.startswith("neg")):
			number_correct += 1

	print "Percent correct = {}, or {}/{}".format(number_correct*100./total_examples, number_correct, total_examples)
	return number_correct*100./total_examples
	
loadWordArrays()
weights=[-7.5, 0.2]
var1 = testSentiment('positiveSentiment.txt', weights)
var2 = testSentiment("negativeSentiment.txt", weights)
print var1+var2
weights=[-6.807843, 0.2]
var1 = testSentiment('positiveSentiment.txt', weights)
var2 = testSentiment("negativeSentiment.txt", weights)
print var1+var2

exit()

"""Below is optimization of the positive and negative weights."""
import scipy.optimize as spo
import numpy as np

def sum_accuracies(weights):
	return -(testSentiment('positiveSentiment.txt', weights) + testSentiment("negativeSentiment.txt", weights))

x0 = np.array([-1., 1.])
x0 = np.array([-7.5, 0.2])
res = spo.minimize(sum_accuracies, x0, method='powell')
print res.x

previous_result = [-7.2303772,0.21397339]