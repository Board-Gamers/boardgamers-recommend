import numpy as np
import pandas as pd
from decouple import Config, RepositoryEnv
from sqlalchemy import create_engine
from scipy.sparse import csr_matrix


# password 보안용
config = Config(RepositoryEnv('boardGameRec/algorithms/.env'))
SQL_PWD = config('MYSQL_PASSWORD')

# SQL 서버와 연결
engine = create_engine(f'mysql://ssafy:{SQL_PWD}@j5a404.p.ssafy.io/boardgamers')
conn_alchemy =  engine.connect()


class MatirxFactorization:

    def __init__(self, R, k, learning_rate, iteration, save_size):
        self.R = R
        self.k = k
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.save_size = save_size


    def matrix_factorization(self):
        result = self.gradient_descent()
        result_df = pd.DataFrame(result, index=self.R.index, columns=self.R.columns)
        self.result_to_sql(result_df)


    def gradient_descent(self):
        user_count, item_count = self.R.shape
        # count*k size의 행렬을 만들어준다
        P = np.random.normal(size=(user_count, self.k))
        Q = np.random.normal(size=(item_count, self.k))

        bu = np.zeros(user_count)
        bi = np.zeros(item_count)

        csr = csr_matrix(self.R.fillna(0))
        ratings = csr.data
        games = csr.indices
        for iter in range(self.iteration):
            for i in range(csr.indptr.size-1):
                for j in range(csr.indptr[i], csr.indptr[i+1]):
                    r = ratings[j]

                    error = self.prediction(P[i, :], Q[games[j], :], bu[i], bi[games[j]]) - r

                    delta_P, delta_bu = self.gradient(error, Q[games[j], :])
                    delta_Q, delta_bi = self.gradient(error, P[i, :])

                    P[i, :] -= delta_P
                    bu[i] -= delta_bu

                    Q[games[j], :] -= delta_Q
                    bi[games[j]] -= delta_bi

            print(iter, self.cost(P, Q, bu, bi))
        
        return P.dot(Q.T) + bu[:, np.newaxis] + bi[np.newaxis:, ]



    def prediction(self, P, Q, bu, bi):
        return P.dot(Q.T) + bu + bi


    def gradient(self, error, weight):
        weight_delta = self.learning_rate * np.dot(weight.T, error)
        bias_delta = self.learning_rate * np.sum(error)
        return weight_delta, bias_delta


    def cost(self, P, Q, bu, bi):
        R = self.R.to_numpy()
        xi, yi = R.nonzero()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(R[x, y] - self.prediction(P[x, :], Q[y, :], bu[x], bi[y]), 2)
        return np.sqrt(cost/len(xi))


    def alternating_least_squares():
        return


    def result_to_sql(self, result_df):
        result_to_save = []
        for i in range(self.R.index.size):
            user_result = result_df.iloc[i]
            user_id = self.R.index[i]

            user_result = user_result.sort_values(ascending=False)
            game_ids = user_result.keys()
            ratings = user_result.values
            for j in range(self.save_size):
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
    if user_rate_limit:
        limit_index = user_rate_count.id.tolist().index(user_rate_limit)
    else:
        limit_index = user_rate_count.size
    user_rate_count = user_rate_count.index[:limit_index]
    most_rated = data.groupby(['game_id']).count().sort_values('id', ascending=False).index[:game_count]
    most_rated = most_rated.to_list()
    
    df = data.pivot_table(index='user_id', columns='game_id', values='rating')

    # most_rated에 속한 game들로만 이루어진 DataFrame 생성
    column_df = pd.DataFrame(index=df.index, columns=['temp'])
    for game in most_rated:
        column_df[game] = df[game]
    column_df.drop('temp', inplace=True, axis=1)
    
    # user_rate_count에 속한 user들로만 이루어진 DataFrame 생성
    if user_rate_limit:
        complete_df = pd.DataFrame(index=['temp'], columns=most_rated)
        for user in user_rate_count:
            complete_df.loc[user] = column_df.loc[user]
        complete_df.drop('temp', inplace=True)
    # user_rate_limit이 0이라면(작성 리뷰 수 제한이 없다면) 따로 DataFrame을 생성하지 않는다
    else:
        complete_df = column_df

    return complete_df.fillna(0)


if __name__ == '__main__':
    csr = make_dataframe(100, 0)
    mf = MatirxFactorization(csr, 3, 0.005, 100, 20)
    mf.matrix_factorization()
