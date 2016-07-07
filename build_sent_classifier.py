import pandas as pd
import nltk, pickle, random, re, sqlite3, time
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC

from nltk.classify import ClassifierI
from statistics import mode

# Use reviews or use tweets
reviews=False

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
		
	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)
	
	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes*1./len(votes)
		return conf

if reviews:
	"""Get review data"""
	conn = sqlite3.connect('knowledgeBase.db')
	conn.text_factory= str
	c = conn.cursor()

	df = pd.read_sql("SELECT * FROM imdb_reviews", conn)
	# Set values for sentiment
	df["sentiment"] = 1.0
	df.ix[df.value<6,"sentiment"] = -1.0
	# Create training data
	documents = [tuple((nltk.word_tokenize(x[0]),x[1])) for x in df[['review','sentiment']].values ]
	# Retrieve all words
	all_words = []
	for doc in documents:
		[all_words.append(word.lower()) for word in doc[0]]
else:
	short_pos = open("positive.txt","r").read()
	short_neg = open("negative.txt","r").read()
	
	documents = []
	for r in short_pos.split("\n"):
		documents.append( (r, "pos") )
	for r in short_neg.split("\n"):
		documents.append( (r, "neg") )
	
	all_words = []
	
	short_pos_words = nltk.word_tokenize(short_pos.decode('utf8'))
	short_neg_words = nltk.word_tokenize(short_neg.decode('utf8'))
	
	for w in short_pos_words:
		all_words.append(w.lower())
	for w in short_neg_words:
		all_words.append(w.lower())

# Sort all words by frequency
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

def find_features(doc):
	words = set(doc)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	
	return features

feature_sets = [(find_features(rev), sent) for rev,sent in documents]
random.shuffle(feature_sets)

doc_rows = len(feature_sets)
train_rows = int(doc_rows*0.6)
cross_rows = train_rows + int(doc_rows*0.2)

training = feature_sets[:train_rows]
cross_val = feature_sets[train_rows:cross_rows]
testing = feature_sets[cross_rows:]

training = feature_sets[:10000]
testing = feature_sets[10000:]
cross_val = testing

print "Length training",len(training)
print "Length cross validation",len(cross_val)
print "Length testing",len(testing)
print "Length of training, cross val, and testing",len(training)+len(cross_val)+len(testing)
print "Number of documents",doc_rows

if reviews:
	"""SIMPLE NAIVE BAYES"""
	try:
		with open("classifiers/naive_bayes.pickle","rb") as classifier_f:
			classifier = pickle.load(classifier_f)
	except IOError, e:
		print str(e)
		with open("classifiers/naive_bayes.pickle","wb") as classifier_f:
			classifier = nltk.NaiveBayesClassifier.train(training)
			pickle.dump(classifier, classifier_f)
	# print "Naive Bayes Algorithm Accuracy Percent:", nltk.classify.accuracy(classifier,cross_val)*100.
	# classifier.show_most_informative_features(15)

	"""MULTINOMIAL NAIVE BAYES"""
	try:
		with open("classifiers/MNB_classifier.pickle","rb") as classifier_f:
			MNB_classifier = pickle.load(classifier_f)
	except IOError, e:
		print str(e)
		with open("classifiers/MNB_classifier.pickle","wb") as classifier_f:
			MNB_classifier = SklearnClassifier(MultinomialNB())
			MNB_classifier.train(training)
			pickle.dump(MNB_classifier, classifier_f)
	# print "Multinomial Naive Bayes Algorithm Accuracy Percent:", nltk.classify.accuracy(MNB_classifier,cross_val)*100.

	"""BERNOULLI NAIVE BAYES"""
	try:
		with open("classifiers/BNB_classifier.pickle","rb") as classifier_f:
			BNB_classifier = pickle.load(classifier_f)
	except IOError, e:
		print str(e)
		with open("classifiers/BNB_classifier.pickle","wb") as classifier_f:
			BNB_classifier = SklearnClassifier(BernoulliNB())
			BNB_classifier.train(training)
			pickle.dump(BNB_classifier, classifier_f)
	# print "Bernoulli Naive Bayes Algorithm Accuracy Percent:", nltk.classify.accuracy(BNB_classifier,cross_val)*100.


	"""LOGISTIC REGRESSION"""
	try:
		with open("classifiers/LogReg_classifier.pickle","rb") as classifier_f:
			LogReg_classifier = pickle.load(classifier_f)
	except IOError, e:
		print str(e)
		with open("classifiers/LogReg_classifier.pickle","wb") as classifier_f:
			LogReg_classifier = SklearnClassifier(LogisticRegression())
			LogReg_classifier.train(training)
			pickle.dump(LogReg_classifier, classifier_f)
	# print "Logistic Regression Algorithm Accuracy Percent:", nltk.classify.accuracy(LogReg_classifier,cross_val)*100.

	"""STOCHASTIC GRADIENT DESCENT"""
	try:
		with open("classifiers/SGD_classifier.pickle","rb") as classifier_f:
			SGD_classifier = pickle.load(classifier_f)
	except IOError, e:
		print str(e)
		with open("classifiers/SGD_classifier.pickle","wb") as classifier_f:
			SGD_classifier = SklearnClassifier(SGDClassifier())
			SGD_classifier.train(training)
			pickle.dump(SGD_classifier, classifier_f)
	# print "SGD Algorithm Accuracy Percent:", nltk.classify.accuracy(SGD_classifier,cross_val)*100.


	"""SUPPORT VECTOR CLASSIFIER"""
	try:
		with open("classifiers/SVC_classifier.pickle","rb") as classifier_f:
			SVC_classifier = pickle.load(classifier_f)
	except IOError, e:
		print str(e)
		with open("classifiers/SVC_classifier.pickle","wb") as classifier_f:
			SVC_classifier = SklearnClassifier(SVC())
			SVC_classifier.train(training)
			pickle.dump(SVC_classifier, classifier_f)
	# print "SUPPORT VECTOR CLASSIFIER Algorithm Accuracy Percent:", nltk.classify.accuracy(SVC_classifier,cross_val)*100.


	"""LINEAR SUPPORT VECTOR CLASSIFIER"""
	try:
		with open("classifiers/LSVC_classifier.pickle","rb") as classifier_f:
			LSVC_classifier = pickle.load(classifier_f)
	except IOError, e:
		print str(e)
		with open("classifiers/LSVC_classifier.pickle","wb") as classifier_f:
			LSVC_classifier = SklearnClassifier(LinearSVC())
			LSVC_classifier.train(training)
			pickle.dump(LSVC_classifier, classifier_f)
	# print "LINEAR SUPPORT VECTOR CLASSIFIER Algorithm Accuracy Percent:", nltk.classify.accuracy(LSVC_classifier,cross_val)*100.

	"""Voted Classifier"""
	try:
		with open("classifiers/Vote_classifier.pickle","rb") as classifier_f:
			Vote_classifier = pickle.load(classifier_f)
	except IOError, e:
		print str(e)
		with open("classifiers/Vote_classifier.pickle","wb") as classifier_f:
			Vote_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, LogReg_classifier, SGD_classifier, SVC_classifier, LSVC_classifier)
			pickle.dump(Vote_classifier, classifier_f)
	print "Vote Classifier Algorithm Accuracy Percent:", nltk.classify.accuracy(Vote_classifier,cross_val)*100.
else:
	try:
		# Retrieve Classifiers
		classifiers = []
		for id_num in range(7):
			with open("classifiers/tweet_clf_{}.pickle".format(id_num), "rb") as clf_file:
				classifiers.append( pickle.load(clf_file) )
	except Exception, e:
		print str(e)	
		#Create classifiers
		classifiers = [
				nltk.NaiveBayesClassifier.train(training),
				SklearnClassifier(MultinomialNB()),
				SklearnClassifier(BernoulliNB()),
				SklearnClassifier(LogisticRegression()),
				SklearnClassifier(SGDClassifier()),
				SklearnClassifier(LinearSVC()),
				SklearnClassifier(SVC())
		]
		with open("classifiers/tweet_clf_6.pickle", "wb") as clf_file:
				pickle.dump(classifiers[0], clf_file)
		id_num = 0
		for clf in classifiers[1:]:
			#Train classifier
			clf.train(training)
			# Save Classifier
			with open("classifiers/tweet_clf_{}.pickle".format(id_num), "wb") as clf_file:
				pickle.dump(clf, clf_file)
			id_num += 1
			print "Classifier Accuracy Percent:", nltk.classify.accuracy(clf,cross_val)*100.

	Vote_classifier = VoteClassifier(*classifiers)
	print "Vote Classifier Algorithm Accuracy Percent:", nltk.classify.accuracy(Vote_classifier,cross_val)*100.
	
