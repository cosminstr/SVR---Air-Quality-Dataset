import pandas as pd

df = pd.read_excel('../datasets/results.xlsx', index_col=0)
df = df.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',  'std_score_time', 
                      'param_C', 'param_epsilon', 'param_gamma', 'param_kernel'])

df.sort_values(by='rank_test_score', ascending=True, inplace=True)
df.to_excel('../datasets/gridSearchResultsOrdered.xlsx')