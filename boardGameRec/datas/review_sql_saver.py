import json
import pymysql
import pandas as pd
from decouple import Config, RepositoryEnv
from sqlalchemy import create_engine

# password 보안용
config = Config(RepositoryEnv('boardGameRec/algorithms/.env'))
SQL_PWD = config('MYSQL_PASSWORD')

# SQL 서버와 연결
engine = create_engine(f'mysql+pymysql://ssafy:{SQL_PWD}@j5a404.p.ssafy.io/boardgamers?charset=utf8', encoding='utf-8')
conn_alchemy =  engine.connect()


# review 데이터를 sql에 저장하는 method
def save_db(gap, portion):
    '''
    gap : 몇 개 row씩 저장할지(chunksize)
    portion : csv에 저장된 전체 1500만개의 데이터 중 저장할 양(ex-0.1, 0.05)
    '''
    # 기존 csv파일에 integer 형식의 user_id가 없어 생성
    with open('boardGameRec/datas/user_ids.json') as json_file:
        ids_json = json.load(json_file)['user_id']
    df = pd.read_csv('boardGameRec/datas/reviews.csv', encoding='utf-8')
    df.columns = ['id', 'user_id', 'rating', 'comment', 'game_id', 'game_name']
    # 필요없는 column 제거
    df.drop(['id', 'comment', 'game_name'], inplace=True, axis=1)
    
    ran = 1 // portion
    for idx, row in df.iterrows():
        if idx and idx % gap == 0:
            save_df = df.iloc[idx-gap:idx-gap+1]
            for i in range(1, gap // ran):
                save_df = save_df.append(df.iloc[idx-gap+ran*i:idx-gap+ran*i+1])
            save_sql(save_df, int(gap*portion), 0)
        name = row['user_id']
        try:
            df.at[idx, 'user_id'] = ids_json[name]
        except:
            df.at[idx, 'user_id'] = -1


# save_db에서 DataFrame을 넘겨받아 sql에 저장하는 method
def save_sql(df, gap, start):
    try:
        df.iloc[start:start+gap].to_sql(
            name='recommend_review', 
            con=engine, 
            if_exists='append', 
            index=False
        )
    except:
        game_exception(df, gap, start)


# sql에 저장 과정 중 오류가 나면 오류가 발생한 row를 제외한 rows를 저장하는 method
def game_exception(df, gap, start):
    gap //= 10
    if gap >= 10:
        for i in range(10):
            save_sql(df, gap, start+gap*i)


if __name__ == '__main__':
    gap = 10000
    portion = 0.01
    save_db(gap, portion)
