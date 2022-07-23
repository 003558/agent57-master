import yaml
import pandas as pd
import datetime

with open('./cnt.yaml', 'r+', encoding='utf-8') as f:
    inp = yaml.safe_load(f)
past       = inp['past']    
TERM       = inp['TERM']
NB_steps   = inp['NB_steps']
start_date = inp['start_date']
flgs       = inp['train_flg']

jis_path     = "./nomura_dam/jisseki.csv"
jis_tmp_path = "./nomura_dam/"
jis_df = pd.read_csv(jis_path)

rain_p_path     = "./nomura_dam/rain_p.csv"
rain_p_tmp_path = "./nomura_dam/"
rain_p_df       = pd.read_csv(rain_p_path)

try_no = 0
count = 0
for flg in flgs:
    count += 1
    if flg:
        term = TERM[try_no]
        nb_steps = NB_steps[try_no]
        Q_tmp = jis_df[(pd.to_datetime(jis_df['datetime'])>=start_date[count]-datetime.timedelta(hours=past)) & \
                       (pd.to_datetime(jis_df['datetime'])<=start_date[count]+datetime.timedelta(minutes=term*2+504*10))]
        Q_tmp.to_csv(jis_tmp_path + "jisseki_tmp_{}.csv".format(try_no), index=False)
        rain_p_tmp = rain_p_df[(pd.to_datetime(rain_p_df['datetime'])>=start_date[count]-datetime.timedelta(hours=past)) & \
                               (pd.to_datetime(rain_p_df['datetime'])<=start_date[count]+datetime.timedelta(minutes=term*2+504*10))]
        rain_p_tmp.to_csv(rain_p_tmp_path + "rain_tmp_p_{}.csv".format(try_no), index=False)
    
        try_no += 1