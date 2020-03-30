import pandas

try:
    raw_data = pandas.read_csv('./datasets/addict-fix.csv')
    N = 1435

    raw_data['PUIcutoff'] = pandas.to_numeric(raw_data['PUIcutoff'], errors='coerce')
    raw_data = raw_data.dropna(subset=['PUIcutoff'])

    # ratio of different values for the column SkolaPoRegionu
    fracs = {'1': 0.143, '2': 0.206, '3': 0.233, '4': 0.323, '5': 0.094}

    groups = raw_data.groupby('PUIcutoff')
    negative = groups.get_group(0.0)
    positive = groups.get_group(1.0)

    Nn = N * len(negative)/len(raw_data)
    Np = N * len(positive)/len(raw_data)

    samplen = pandas.concat(dff.sample(n=int(fracs.get(str(i)) * Nn)) for i, dff in negative.groupby('SkolaPoRegionu'))
    samplep = pandas.concat(dff.sample(n=int(fracs.get(str(i)) * Np)) for i, dff in positive.groupby('SkolaPoRegionu'))

    result = pandas.concat([samplen, samplep])

    raw_data.to_csv('./datasets/addict-fix-limit.csv')

except Exception as err:
    print(str(err))