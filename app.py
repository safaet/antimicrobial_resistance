
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score
from sklearn.metrics import auc
import sklearn.metrics

from fastapi.middleware.cors import CORSMiddleware

clf1 = LogisticRegression(random_state=42)
clf2 = XGBClassifier(random_state=42)
clf3 = SVC(probability=True, random_state=42)

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
xgb_model = XGBClassifier(random_state=42)

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

models.append(('ENS-1', VotingClassifier(
    estimators=[
        ('log_reg', log_reg_model),
        ('gb', gb_model),
        ('nn', nn_model),
        ('xgb', xgb_model)
    ],
    voting='soft'  # Use 'hard' for majority voting or 'soft' for weighted voting
)))

models.append(('ENS-2', VotingClassifier(estimators=[
    ('lr', clf1), ('rf', clf2), ('svc', clf3)
],voting='soft'
)))

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
    X = training_data.iloc[:, 1:-1]
    y = training_data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    li = []

    for name, model in models:
        model = model.fit(X_train, y_train)
        
        Y_train_pred = model.predict(X_train)
        Tr_precision = precision_score(y_train, Y_train_pred, average='macro').round(3)
        
        Tr_recall = recall_score(y_train, Y_train_pred, average='macro').round(3)

        
        Tr_f1 = f1_score (y_train, Y_train_pred, average='macro').round(3)
        
        accuracy = model.score(X_train, y_train).round(3)
        

        cv_score = cross_val_score(model, X,y, cv=6)
        mean_accuracy = sum(cv_score)/len(cv_score)
        mean_accuracy = round(mean_accuracy, 2)
        # tr_cv.append(mean_accuracy)


        Y_test_pred = model.predict(X_test)
        report = sklearn.metrics.classification_report(y_test, Y_test_pred)
        Te_precision = precision_score(y_test, Y_test_pred, average='macro').round(3)
        
        Te_recall = recall_score(y_test, Y_test_pred, average='macro').round(3)
        
        Te_f1 = f1_score (y_test, Y_test_pred, average='macro').round(3)
        
        accuracy = model.score(X_test, y_test).round(3)

        cv_score = cross_val_score(model, X,y, cv=6)
        mean_accuracy = sum(cv_score)/len(cv_score)
        mean_accuracy = round(mean_accuracy, 2)


        dic = {f'classifier': name, 'tr_precision': Tr_precision, 'tr_recall': Tr_recall, 'tr_f1': Tr_f1,'tr_accuracy': accuracy, 'tr_cv': mean_accuracy,
         'te_precision': Te_precision, 'te_recall': Te_recall, 'te_f1': Te_f1, 'te_accuracy': accuracy, 'te_cv': mean_accuracy}
        
        li.append(dic)

    # dict = {'classifier': mode, 'tr_precision': tr_precision, 'tr_recall': tr_recall, 'tr_f1 ': tr_f1,'tr_accuracy': tr_accu, 'tr_cv': tr_cv,
    #       'te_precision': te_precision, 'te_recall': te_recall, 'te_f1 ': te_f1, 'te_accuracy': te_accu, 'te_cv': te_cv}

    return li


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
