import pandas
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    # loading raw data from dataset
    raw_data = pandas.read_csv('./datasets/addict.csv')

    # convert everything to numeric data
    # missing value are replaced with NoN
    for column in raw_data.columns:
        raw_data[column] = pandas.to_numeric(raw_data[column], errors='coerce')

    # drop all rows where PUIcutoof column is NoN
    raw_data = raw_data.dropna(subset=['PUIcutoff'])

    # extracting only feature that we want to use for binary classification
    # for this pass we don't want to use TEMPS features that were used to determine subjects temperaments
    feature_names = [
        column for column in raw_data.columns
        if not column.startswith('TEMPS')
            and not column.startswith('Internet')
            and column not in ['ID', 'FBupotreba', 'PROT_SADR_AKT', 'RISK_SADR_AKT', 'Temper_bin', 'NKP', 'PI', 'SPO', 'PUI', 'PUIcutoff']
    ]

    # class column for binary classification of addiction                 ]
    class_name = 'PUIcutoff'

    feature_data_raw = raw_data[feature_names]
    class_data = raw_data[class_name]

    # preprocessing data

    imputer = SimpleImputer(missing_values=numpy.nan, strategy='most_frequent')
    imputed_data = imputer.fit_transform(feature_data_raw)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    feature_data_processed = pandas.DataFrame(scaled_data)
    feature_data_processed.columns = feature_data_raw.columns

    # split data to trainig and test set (30% test set)
    train_feature_data, test_feature_data, train_class_data, test_class_data = \
        train_test_split(feature_data_processed, class_data, test_size=0.3, stratify=class_data)

    # classification using Decision Tree
    print('Decision Tree classifier')
    # create model using training data
    classifier = DecisionTreeClassifier()
    classifier.fit(train_feature_data, train_class_data)

    # predict using model and test data
    test_predicted_data = classifier.predict(test_feature_data)

    # calculate metrics
    print('score={}'.format(accuracy_score(test_class_data, test_predicted_data)))
    print(confusion_matrix(test_class_data, test_predicted_data))
    print(classification_report(test_class_data, test_predicted_data))

    # classification using SVM
    print('SVM classifier')
    # create model using training data
    classifier = SVC(kernel='linear')
    classifier.fit(train_feature_data, train_class_data)

    # predict using model and test data
    test_predicted_data = classifier.predict(test_feature_data)

    # calculate metrics
    print('score={}'.format(accuracy_score(test_class_data, test_predicted_data)))
    print(confusion_matrix(test_class_data, test_predicted_data))
    print(classification_report(test_class_data, test_predicted_data))

    # classification using Random Forest
    print('Random Forest classifier')

    # create model using training data
    classifier = RandomForestClassifier(n_estimators=100,
                                        random_state=0,
                                        max_features='sqrt',
                                        n_jobs=-1, verbose=1)
    classifier.fit(train_feature_data, train_class_data)

    # predict using model and test data
    test_predicted_data = classifier.predict(test_feature_data)

    # calculate metrics
    print('score={}'.format(accuracy_score(test_class_data, test_predicted_data)))
    print(confusion_matrix(test_class_data, test_predicted_data))
    print(classification_report(test_class_data, test_predicted_data))

    # classification using LogisticRegression
    print('Logistic Regression classifier')
    # create model using training data
    classifier = LogisticRegression(max_iter=10000, solver='lbfgs')
    classifier.fit(train_feature_data, train_class_data)

    # predict using model and test data
    test_predicted_data = classifier.predict(test_feature_data)

    # calculatre metrics
    print('score={}'.format(accuracy_score(test_class_data, test_predicted_data)))
    print(confusion_matrix(test_class_data, test_predicted_data))
    print(classification_report(test_class_data, test_predicted_data))

    # using RFE (Recursive Feature Estimation)
    print('Logistic Regression classifier after using RFE')
    classifier = LogisticRegression(max_iter=10000, solver='lbfgs')
    rfe = RFE(classifier, 5)

    rfe_data = rfe.fit_transform(feature_data_processed, class_data)
    selected_columns = rfe.get_support(indices=True)

    columns_rfe = [
        feature_data_processed.columns[selected] for selected in selected_columns
    ]

    feature_data_rfe = pandas.DataFrame(rfe_data)
    feature_data_rfe.columns = columns_rfe

    # split rfe data to trainig and test set (30% test set)
    train_feature_data, test_feature_data, train_class_data, test_class_data = \
        train_test_split(feature_data_rfe, class_data, test_size=0.3, stratify=class_data)

    classifier.fit(train_feature_data, train_class_data)

    # predict using model and test data
    test_predicted_data = classifier.predict(test_feature_data)

    # calculatre metrics
    print('score={}'.format(accuracy_score(test_class_data, test_predicted_data)))
    print(confusion_matrix(test_class_data, test_predicted_data))
    print(classification_report(test_class_data, test_predicted_data))

except Exception as ex:
    print(str(ex))
