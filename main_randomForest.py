import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from preProc import procData


def gridSearch(X, y, predName):

    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    results = np.empty([0, 3])

    for train_index, test_index in cv.split(X, y):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for n_estimators in range(50,550,50):
            for max_depth in range(1,16):
        
                mdl = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,  n_jobs=8, class_weight='balanced')
                mdl.fit(X_train, y_train)
                
                pred = mdl.predict(X_test)
                auc = metrics.roc_auc_score(y_test,pred)
                
                output = np.array([n_estimators,max_depth, auc])
                results = np.vstack((results, output))
    
    n_estimators = results[np.argmax(results[:,2]),0]
    max_depth = results[np.argmax(results[:,2]),1]
    auc = results[np.argmax(results[:,2]),2]
    return auc, int(n_estimators), int(max_depth)

def crossVal(X, y, predName, n_estimators, max_depth):

    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    aucResults = []
    f1Results = []

    for train_index, test_index in cv.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        mdl = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,  n_jobs=8, class_weight='balanced')
        mdl.fit(X_train, y_train)
        
        pred = mdl.predict(X_test)
        auc = metrics.roc_auc_score(y_test,pred)
        f1 = metrics.f1_score(y_test,pred)
        aucResults.append(auc)
        f1Results.append(f1)

    print('AUC mean=',np.mean(aucResults), 'std=',np.std(aucResults))


X, Y_churn, Y_appetency, Y_upselling = procData(50000, 0.3, 100)

print('Training models...')
auc, n_estimators, max_depth = gridSearch(X, Y_churn, 'churn')
print('churn - best random forest model: AUC=',auc,'n_estimators=',n_estimators,'max_depth=',max_depth)
crossVal(X, Y_churn, 'churn', n_estimators, max_depth)

auc, n_estimators, max_depth = gridSearch(X, Y_appetency, 'appetency')
print('appetency - best random forest model: AUC=',auc,'n_estimators=',n_estimators,'max_depth=',max_depth)
crossVal(X, Y_appetency, 'appetency', n_estimators, max_depth)

auc, n_estimators, max_depth = gridSearch(X, Y_upselling, 'up-selling')
print('up-selling - best random forest model: AUC=',auc,'n_estimators=',n_estimators,'max_depth=',max_depth)
crossVal(X, Y_upselling, 'up-selling', n_estimators, max_depth)



