import csv
import json
import MySQLdb
import pymysql
import numpy as np
import pandas as pd
from decouple import Config, RepositoryEnv
from sqlalchemy import create_engine

# password 보안용
config = Config(RepositoryEnv('boardGameRec/algorithms/.env'))
SQL_PWD = config('MYSQL_PASSWORD')

db = pymysql.connect(user='ssafy', host='j5a404.p.ssafy.io', passwd=SQL_PWD, port=3306, db='boardgamers')

# SQL 서버와 연결
engine = create_engine(f'mysql+pymysql://ssafy:{SQL_PWD}@j5a404.p.ssafy.io/boardgamers?charset=utf8', encoding='utf-8')
conn_alchemy =  engine.connect()


def save_sql():
    with open('boardGameRec/datas/user_ids.json') as json_file:
        ids_json = json.load(json_file)['user_id']
    df = pd.read_csv('boardGameRec/datas/reviews.csv', encoding='utf-8')
    df.columns = ['id', 'user_id', 'rating', 'comment', 'game_id', 'game_name']
    
    gap = 10000
    for idx, row in df.iterrows():
        if idx and idx % gap == 0:
            df.iloc[(idx//gap-1)*gap:(idx//gap)*gap].to_sql(name='test_review', con=engine, if_exists='append', index=False)
            print(idx)
        name = row['user_id']
        try:
            df.at[idx, 'user_id'] = ids_json[name]
        except:
            df.at[idx, 'user_id'] = 0


if __name__ == '__main__':
    save_sql()
