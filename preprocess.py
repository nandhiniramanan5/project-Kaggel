import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import decomposition

# Reads data and preprocesses it to fit our data structures. These will be input to our models
def getTrainData(filename, test = False):
    f = open(filename,'r')
    count = 0
    testData, trainData, trainLabels = [], [], []
    for line in f:
        if count == 0:
            count += 1
            continue
        if count == 3000 and not test:
            break
        if count == 300 and test:
            break
        arr = line.split(',')
        if not test:
            if len(arr) != 785:
                print arr
            trainLabels.append(int(arr[0]))
            trainData.append(map(float, arr[1:]))
        else:
            if len(arr) != 784:
                print len(arr)
            testData.append(map(float, arr))
        count += 1
    f.close()
    if not test:
        return (trainLabels, trainData)
    return testData

# Benchmark results obtained to compare our model
def getBenchMarkTestLabels(filename):
    f = open(filename, 'r')
    trainLabels = []
    count = 0
    for line in f:
        if count == 0:
            count += 1
            continue
        if count == 300:
            break
        trainLabels.append(int(line.split(',')[1].strip()))
        count += 1
    f.close()
    return trainLabels

# Trains the model given as input preference with the train data and labels
def getModel(model, data, labels, param):
    trainModel = None
    if model == "knn":
        trainModel = KNeighborsClassifier(n_neighbors=23)
    if model == "rforest":
        trainModel = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
    trainModel.fit(data, labels)
    return trainModel

# Classifies the test data and produces the accuracy for the model
def classify(model, data, labels = None):
    out = []
    for i in range(len(data)):
        inp = np.array(data[i]).reshape((1, -1))
        ans = model.predict(inp)
        out.append(ans[0])
    print "Accuracy: ", accuracy_score(labels, out)

# Reduce the dimensions to extract the best features from the data set (Currently set to 100)
def reduceDimensions(data, nDim = 200):
    pca = decomposition.PCA(nDim)
    pca.fit(data)
    return pca.transform(data)

# Shuffles the data using the indices and does cross validation
def getShuffledData(data, labels):
    samples = np.array(data).shape[0]
    # Shuffle the indices in a random order and give around 6 splits
    # In each of the split, the test data is 0.3 times the size of the entire data (70-30) split
    # This is a 7 fold cross validation
    cv = cross_validation.ShuffleSplit(samples, n_iter=7, test_size=0.3, random_state=0)
    return cv

# Does Crossvalidation and returns the best hyperparameter for the model.
def crossValidation(data, labels, model):
    cv = getShuffledData(data, labels)
    knn = []
    if model == "knn":
        knn = [23, 41, 13, 53, 79]
    if model == "rforest":
        knn = [10, 26, 50, 76, 100]
    temp = float(0)
    best = 0
    clf = None
    for i in knn:
        if model == "knn":
            clf = KNeighborsClassifier(n_neighbors=i)
        if model == "rforest":
            clf = RandomForestClassifier(n_estimators=i, max_depth=None, min_samples_split=1, random_state=0)
        scores = cross_validation.cross_val_score(clf, data, labels, cv=cv)
        if scores.mean() > temp:
            temp = scores.mean()
            best = i
    return best


def trainAndTestModel(trainData, trainLabels, newTrainData, testData, testLabels, model):
    print "Best Hyper Parameter with reduced Dimensionality",
    # Do Cross Validation on the reduced Dimesional Data to get the best hyperparameter
    neigh = crossValidation(newTrainData, trainLabels, model=model)
    print neigh
    print "Prediction ",
    # Train the model with the best hyper parameter
    clf = getModel(model, trainData, trainLabels, neigh)
    # Do classification on the data
    classify(clf, testData, testLabels)
    print "------------------------------------------------"
    print "Best Hyper Parameter with original Dimensionality",
    # Do Cross Validation on the original Dimensional Data to get the best hyperparameter
    neigh = crossValidation(trainData, trainLabels, model)
    print neigh
    print "Prediction ",
    # Train with the best hyper parameter
    clf = getModel(model, trainData, trainLabels, neigh)
    # Do Classification
    classify(clf, testData, testLabels)
    print "------------------------------------------------"

if __name__ == "__main__":
    # Get the train data and the labels
    trainLabels, trainData = getTrainData('train.csv')
    # Get the test data and the labels
    testData = getTrainData('test.csv', True)
    testLabels = getBenchMarkTestLabels('submission.csv')
    # Get the reduced Dimensional data (Here the dimension is reduced to 100) --> You can change it further
    newTrainData = reduceDimensions(trainData)
    # Train and test the model using KNN
    trainAndTestModel(trainData, trainLabels, newTrainData, testData, testLabels, "knn")
    # Train and test the model using RandomForest
    trainAndTestModel(trainData, trainLabels, newTrainData, testData, testLabels, "rforest")