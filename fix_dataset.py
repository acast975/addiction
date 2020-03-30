import pandas

try:
    # loading raw data from dataset
    raw_data = pandas.read_csv('./datasets/addict.csv')

    # iterate through all columns
    for column in raw_data.columns:
        if column in ['ImaKomp', 'DaMozeDaLiBi', 'EnergetskoP', 'Grickalice', 'Pusac', 'Droga', 'Pol']:
            raw_data[column].replace({'2': '0'}, inplace=True)

        if column == 'Uspeh':
            raw_data[column].replace({'1': '4', '2': '3', '3': '2', '4': '1'}, inplace=True)

        if column == 'Predhodnih6meseci':
            raw_data[column].replace({'2': '0', '3': '1', '1': '2'}, inplace=True)

        if column == 'DaLiSvakodnevnoFb':
            raw_data[column].replace({'2': '1', '1': '2'}, inplace=True)

    raw_data.rename(columns={'BrojDana': 'BrojDanaFizickeAktivnosti'})

    raw_data.to_csv('./datasets/addict-fix.csv', index=False)
except Exception as err:
    print(str(err))
