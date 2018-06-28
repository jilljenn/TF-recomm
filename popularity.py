import dataio
import numpy as np


df_train, _, df_test = dataio.get_data()
count_work = df_train.groupby('item').size().sort_index() #.to_frame('count_user').sort_values('item')
print(count_work.head())
work_ids = count_work.index
work_count = count_work.values
print(len(work_ids), 'works until', work_ids.max())

popularity = np.zeros(len(work_ids))
popularity[work_ids] = work_count
print(popularity[:5])
#print(count_work.index[:5][1:3].argmin())
