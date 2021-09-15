import MySQLdb
import numpy as np
import pandas as pd
from decouple import Config, RepositoryEnv
from sqlalchemy import create_engine
from scipy.sparse.linalg import svds


# password 보안용
config = Config(RepositoryEnv('boardGameRec/algorithms/.env'))
SQL_PWD = config('MYSQL_PASSWORD')

# SQL 서버와 연결
engine = create_engine(f'mysql://ssafy:{SQL_PWD}@j5a404.p.ssafy.io/boardgamers')
conn_alchemy =  engine.connect()


def model_based_svd(user_name):
    sql_data = pd.read_sql('SELECT user, rating, game_id FROM review WHERE id<10000 AND game_id<300', conn_alchemy)
    df_ratings = sql_data.pivot(index='user', columns='game_id', values='rating').fillna(0)
    df_svd = make_df_svd(df_ratings)
    sorted_predictions = df_svd.loc[user_name].sort_values(ascending=False)
    seen_movies = df_ratings.loc[user_name]
    for i in range(seen_movies.size):
        if float(seen_movies.iloc[i]):
            sorted_predictions = sorted_predictions.drop(seen_movies.index[i])
    print(sorted_predictions)
    

# model based의 SVD(Singular Value Decomposition)을 활용한 Matrix를 만드는 method
def make_df_svd(df):
    matrix = df.to_numpy()
    df_mean = np.mean(matrix, axis=1)
    matrix_mean = matrix - df_mean.reshape(-1, 1)
    U, sigma, Yt = svds(matrix_mean, k=3)
    sigma = np.diag(sigma)
    svd_predicted = np.dot(np.dot(U, sigma), Yt) + df_mean.reshape(-1, 1)
    return pd.DataFrame(svd_predicted, index=df.index, columns=df.columns)


if __name__ == '__main__':
    # model_based_svd('zeriphon')
    df = pd.read_sql('SELECT user, rating, game_id FROM review', conn_alchemy).sort_values(by='game_id')
    for i in range((df.size // (3*1000000))+1):
        limit = i*1000000
        if i:
            df_pivot_limit = df[limit:limit+1000000].pivot(index='user', columns='game_id', values='rating')
            df_pivot = pd.concat([df_pivot, df_pivot_limit], axis=1)
        else:
            df_pivot = df[limit:limit+1000000].pivot(index='user', columns='game_id', values='rating')
        print(df_pivot)
    # df_pivot = df.groupby(['user', 'game_id'])['rating'].max().unstack().fillna(0)
    # print(df_pivot)
    # print('pivoting 완료')
    # df_pivot.to_csv('user_rating.csv', mode='w')


## MySQLdb 사용법
# conn = MySQLdb.connect(host='j5a404.p.ssafy.io', user='ssafy', passwd='qweasd123*', db='boardgamers')
# cursor = conn.cursor()
# cursor.execute('SELECT * FROM boardgame LIMIT 10')
# for game in cursor.fetchall():
#     print(game)