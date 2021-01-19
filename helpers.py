import pandas
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(dataset_file_path):
    """

    :param dataset_file_path:
    :return:
    """
    raw_data = None
    try:
        # loading raw data from dataset
        raw_data = pandas.read_csv(dataset_file_path)

        # convert everything to numeric data
        # missing value are replaced with NoN
        for column in raw_data.columns:
            raw_data[column] = pandas.to_numeric(raw_data[column], errors='coerce')

        # drop all rows where PUIcutoof column is NoN
        raw_data = raw_data.dropna(subset=['PUIcutoff'])

        # extracting only columns we want to use for further data analysus
        column_names = [
            column for column in raw_data.columns
            if column not in ['ID', 'NKP', 'PI', 'SPO', 'PUI', 'PUIcutoff', 'CutOff4950']
        ]
        raw_data = raw_data[column_names]
    except Exception as err:
        print(str(err))
    finally:
        return raw_data


def load_clean_data(dataset_file_path, class_name):
    """

    :param dataset_file_path: str
    :param class_name: str
    :return:
    """

    feature_data = None
    class_data = None
    try:
        # loading raw data from dataset
        raw_data = pandas.read_csv(dataset_file_path)

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
                and not column.startswith('Sadrzaj')
                and not column.startswith('Aktivnost')
                and column not in ['ID', 'FBupotreba', 'KolikoCigareta', 'NeZnaNet', 'PROT_SADR_AKT', 'RISK_SADR_AKT', 'Temper_bin', 'NKP', 'PI', 'SPO', 'PUI', 'PUIcutoff', 'CutOff3940', 'CutOff4950']
            # if column.startswith('Internet')
        ]

        # extracting data (both features and class)
        feature_data = raw_data[feature_names]
        class_data = raw_data[class_name]

    except Exception as ex:
        print(str(ex))
    finally:
        return feature_data, class_data


def preprocess_data(feature_data):
    """

    :param feature_data:
    :return:
    """
    feature_data_processed = None

    try:
        imputer = SimpleImputer(missing_values=numpy.nan, strategy='most_frequent')
        imputed_data = imputer.fit_transform(feature_data)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_data)

        feature_data_processed = pandas.DataFrame(scaled_data)
        feature_data_processed.columns = feature_data.columns
    except Exception as ex:
        print(str(ex))
    finally:
        return feature_data_processed

