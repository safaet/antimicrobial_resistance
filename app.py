from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import tempfile

app = FastAPI()

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import auc

models = []
models.append(('LogR', LogisticRegression()))
models.append(('gNB', GaussianNB()))
models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('mNB', MultinomialNB()))
models.append(('ABC', AdaBoostClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('BC', BaggingClassifier()))

# def train_model():
#     # Assuming you have a pre-existing CSV file for training the model
#     training_data = pd.read_csv('BankNote_Authentication.csv')
#     X_train = training_data.iloc[:, :-1]
#     y_train = training_data.iloc[:, -1]
#     model.fit(X_train, y_train)

# train_model()

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    # Check if the uploaded file is a CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    # Load the CSV file
    try:
        data = pd.read_csv(temp_file_path)
    except Exception as e:
        return HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
    finally:
        # Clean up temporary file
        temp_file.close()

    training_data = pd.read_csv(temp_file_path)
    X_train = training_data.iloc[:, 1:-1]
    y_train = training_data.iloc[:, -1]


    mode = []
    tr_precision=[]
    tr_recall = []
    tr_f1 = []
    tr_accu = []
    tr_cv = []

    te_precision=[]
    te_recall = []
    te_f1 = []
    te_accu = []
    te_cv = []
    

    for name, model in models:
        model = model.fit(X_train, y_train)
        mode.append(name)
        
        Y_train_pred = model.predict(X_train)
        Tr_precision = precision_score(y_train, Y_train_pred, average='macro').round(3)
        tr_precision.append(Tr_precision)
        
        Tr_recall = recall_score(y_train, Y_train_pred, average='macro').round(3)
        tr_recall.append(Tr_recall)
        
        Tr_f1 = f1_score (y_train, Y_train_pred, average='macro').round(3)
        tr_f1.append(Tr_f1)
        
        accuracy = model.score(X_train, y_train).round(3)
        tr_accu.append(accuracy)

        # cv_score = cross_val_score(model, X,y, cv=6)
        # mean_accuracy = sum(cv_score)/len(cv_score)
        # mean_accuracy = round(mean_accuracy, 2)
        # # tr_cv.append(mean_accuracy)

    dict = {'classifier': mode, 'tr_precision': tr_precision, 'tr_recall': tr_recall, 'tr_f1 ': tr_f1,'tr_accuracy': tr_accu, 'tr_cv': tr_cv,
         'te_precision': te_precision, 'te_recall': te_recall, 'te_f1 ': te_f1, 'te_accuracy': te_accu, 'te_cv': te_cv}

    return dict