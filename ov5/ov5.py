from sklearn.naive_bayes import BernoulliNB
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
#from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sklearn

vectorizer = HashingVectorizer(binary=True, stop_words='english')
#unpacking
data = pickle.load(open('sklearn-data.pickle','rb'))
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

#hashingvecotrizing
x_train = vectorizer.fit_transform(x_train) 
x_test = vectorizer.fit_transform(x_test)

#training
# classifier_b = BernoulliNB(alpha=0.1) #wikipedia sier at mange mener denne burde være 1, dette gir en accuracy score på 83 % i min testing
# #0.1 derimot gir bedre score og bruker dermed den.
# b = classifier_b.fit(X=x_train,y=y_train)
# pred_b = classifier_b.predict(x_test)
# score_b = accuracy_score(y_test,pred_b)
#print(score) #score med alpha=0.1 => 85%

print('0')
clf = DecisionTreeClassifier()
print('1')
clf = clf.fit(x_train,y_train)
print('2')
pred_t = clf.predict(x_test)
print('3')
score_t = accuracy_score(y_test,pred_t)
print('4')
print(score_t)


