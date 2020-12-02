import matplotlib.pyplot as plot
import seaborn as sns
import helpers
import pandas

feature_data = helpers.load_clean_data('./datasets/addict-fix-limit.csv')
feature_data_count = len(feature_data)

for feature in feature_data:
    df = pandas.DataFrame({'Counts': feature_data.groupby([feature]).size()})
    df['Percent'] = (df['Counts'] / feature_data_count) * 100
    df1 = feature_data.groupby([feature, 'PUIcutoff']).size().unstack()
    df1.insert(1, '0.0%', (df1[0] / feature_data_count) * 100)
    df1.insert(3, '1.0%', (df1[1] / feature_data_count) * 100)
    print(df1[:5])
