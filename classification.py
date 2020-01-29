import pandas
import numpy
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer

try:
    # loading raw data from dataset
    raw_data = pandas.read_csv('./datasets/addict.csv')

    # extracting only feature that we want to use for binary classification
    # for this pass we don't want to use TEMPS features that were used to determine subjects temperaments
    feature_names = [
        column for column in raw_data.columns
        if not column.startswith('TEMPS')
            and column not in ['ID', 'FBupotreba', 'PROT_SADR_AKT', 'RISK_SADR_AKT', 'NKP', 'PI', 'SPO', 'PUI']
    ]

    # class column for binary classification of addiction                 ]
    class_name = 'PUIcutoff'

    feature_data_raw = raw_data[feature_names]
    class_data = raw_data[class_name]

    # preprocessing data

    # convert everything to numeric data
    # missing value are replaced with NoN
    for column in feature_data_raw.columns:
        feature_data_raw[column] = pandas.to_numeric(feature_data_raw[column], errors='coerce')

    imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    imputed_data = imputer.fit_transform(feature_data_raw)

    normalizer = Normalizer(norm='l2')
    normalized_data = normalizer.fit_transform(imputed_data)

    feature_data_processed = pandas.DataFrame(normalized_data)
    feature_data_processed.columns = feature_data_raw.columns
    print('Hello world!')
except Exception as ex:
    print(str(ex))
