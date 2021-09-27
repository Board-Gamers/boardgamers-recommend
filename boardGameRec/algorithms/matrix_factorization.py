import numpy as np
import pandas as pd
from decouple import Config, RepositoryEnv
from sqlalchemy import create_engine


# password 보안용
config = Config(RepositoryEnv('boardGameRec/algorithms/.env'))
SQL_PWD = config('MYSQL_PASSWORD')

# SQL 서버와 연결
engine = create_engine(f'mysql://ssafy:{SQL_PWD}@j5a404.p.ssafy.io/boardgamers')
conn_alchemy =  engine.connect()


def matrix_factorization(R):
    k = 3
    iteration = 3
    learning_rate = 0.005
    result = gradient_descent(R, k, iteration, learning_rate)
    result_df = pd.DataFrame(result, index=R.index, columns=R.columns)
    result_to_sql(result_df, R, 20)


def gradient_descent(R, k, iteration, learning_rate):
    user_count, item_count = R.shape
    # count*k size의 행렬을 만들어준다
    P = np.random.normal(size=(user_count, k))
    Q = np.random.normal(size=(item_count, k))

    bu = np.zeros(user_count)
    bi = np.zeros(item_count)

    for iter in range(iteration):
        for u in range(user_count):
            for i in range(item_count):
                r = R.iloc[u].iloc[i]
                if r >= 0:
                    error = prediction(P[u, :], Q[i, :], bu[u], bi[i]) - r

                    delta_P, delta_bu = gradient(error, Q[i, :], learning_rate)
                    delta_Q, delta_bi = gradient(error, P[u, :], learning_rate)

                    P[u, :] -= delta_P
                    bu[u] -= delta_bu

                    Q[i, :] -= delta_Q
                    bi[i] -= delta_bi

    return P.dot(Q.T) + bu[:, np.newaxis] + bi[np.newaxis:, ]



def prediction(P, Q, bu, bi):
    return P.dot(Q.T) + bu + bi


def gradient(error, weight, learning_rate):
    weight_delta = learning_rate * np.dot(weight.T, error)
    bias_delta = learning_rate * np.sum(error)
    return weight_delta, bias_delta



def result_to_sql(result_df, original_df, save_size):
    result_to_save = []
    for i in range(original_df.index.size):
        user_result = result_df.iloc[i]
        user_id = original_df.index[i]

        user_result = user_result.sort_values(ascending=False)
        game_ids = user_result.keys()
        ratings = user_result.values
        for j in range(save_size):
            result_to_save.append([user_id, game_ids[j], j+1, ratings[j]])

    df_to_save = pd.DataFrame(result_to_save, columns=['user_id', 'game_id', 'rank', 'predicted_rating'])

    df_to_save.to_sql(
        name='recommend_result',
        con=engine,
        if_exists='append',
        index=False
    )


'''
일부 데이터만 활용하는 DataFrame 제작
game_count : 평점 수를 기준으로 상위 x개만 알고리즘 실행
user_rate_limit : 해당 개수 이하의 평점을 매긴 user는 제외
'''
def make_dataframe(game_count, user_rate_limit):
    query = 'SELECT * FROM boardgamers.recommend_review LIMIT 10000;'
    data = pd.read_sql(query, conn_alchemy)

    user_rate_count = data.groupby(['user_id']).count().sort_values('id', ascending=False)
    limit_index = user_rate_count.id.tolist().index(user_rate_limit)
    user_rate_count = user_rate_count.index[:limit_index]
    most_rated = data.groupby(['game_id']).count().sort_values('id', ascending=False).index[:game_count]

    df = data.pivot_table(index='user_id', columns='game_id', values='rating')
    for user in df.index:
        if user not in user_rate_count:
            df.drop(user, inplace=True)
    for game in df.columns:
        if game not in most_rated:
            df.drop(game, inplace=True, axis=1)
    return df


if __name__ == '__main__':
    R = make_dataframe(30, 1)
    matrix_factorization(R)
