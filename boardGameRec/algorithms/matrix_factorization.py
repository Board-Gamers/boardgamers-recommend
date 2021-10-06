import numpy as np
import pandas as pd
from decouple import Config, RepositoryEnv
from sqlalchemy import create_engine
from scipy.sparse import csr_matrix
import pymysql
pymysql.install_as_MySQLdb()


class MatirxFactorization:

    def __init__(self, csr, k, learning_rate, iteration, save_size, engine, matrix):
        self.csr = csr
        self.k = k
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.save_size = save_size
        self.engine = engine
        self.matrix = matrix


    def matrix_factorization(self):
        self.gradient_descent()
        # result = self.gradient_descent()
        # result_df = pd.DataFrame(result, index=self.R.index, columns=self.R.columns)
        # self.result_to_sql(result_df)


    def gradient_descent(self):
        user_count, item_count = len(self.matrix[0]), len(self.matrix[1])
        # count*k size의 행렬을 만들어준다
        P = np.random.normal(size=(user_count, self.k))
        Q = np.random.normal(size=(item_count, self.k))

        bu = np.zeros(user_count)
        bi = np.zeros(item_count)

        csr = self.csr
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

            print(iter)
        
        self.result_to_sql(P, Q, bu, bi)

        #     print(iter, self.cost(P.dot(Q.T) + bu[:, np.newaxis] + bi[np.newaxis:, ]))

        # return P.dot(Q.T) + bu[:, np.newaxis] + bi[np.newaxis:, ]


    def prediction(self, P, Q, bu, bi):
        return P.dot(Q.T) + bu + bi


    def gradient(self, error, weight):
        weight_delta = self.learning_rate * np.dot(weight.T, error)
        bias_delta = self.learning_rate * np.sum(error)
        return weight_delta, bias_delta


    # def cost(self, prediction):
    #     R = self.R.to_numpy()
    #     xi, yi = R.nonzero()
    #     cost = 0
    #     for x, y in zip(xi, yi):
    #         cost += pow(R[x, y] - prediction[x, y], 2)
    #     return np.sqrt(cost/len(xi))


    def result_to_sql(self, P, Q, bu, bi):
        result_to_save = []

        # BoardGameGeek 데이터가 아닌 실제 유저 id는 1000000부터 시작하므로 그 인덱스를 찾는다
        user_start = 1000000
        while True:
            try:
                user_index = self.matrix[0].index(user_start)
                break
            except:
                user_start += 1

        for i in range(user_index, len(self.matrix[0])):
            user_result = self.prediction(P[i, :], Q, bu[i], bi)
            user_id = self.matrix[0][i]
            
            games_rated = []
            for j in range(self.csr.indptr[i], self.csr.indptr[i+1]):
                games_rated.append(self.matrix[1][self.csr.indices[j]])

            user_result = pd.Series(user_result, self.matrix[1]).sort_values(ascending=False)
            
            game_ids = user_result.keys()
            ratings = user_result.values

            saved = 0
            game_ids_idx = 0
            while saved < self.save_size:
                save_game_id = game_ids[game_ids_idx]
                if save_game_id not in games_rated:
                    result_to_save.append([user_id, game_ids[game_ids_idx], game_ids_idx+1, ratings[game_ids_idx]])
                    saved += 1
                game_ids_idx += 1

        df_to_save = pd.DataFrame(result_to_save, columns=['user_id', 'game_id', 'rank', 'predicted_rating'])

        df_to_save.to_sql(
            name='recommend_result',
            con=self.engine,
            if_exists='append',
            index=False
        )


'''
일부 데이터만 활용하는 DataFrame 제작
game_count : 평점 수를 기준으로 상위 x개만 알고리즘 실행
'''
def make_csr(game_count, con):
    query = 'SELECT * FROM boardgamers.recommend_review;'
    data = pd.read_sql(query, con)

    users = data.groupby(['user_id']).count().index.to_list()
    
    game_rate_count = data.groupby(['game_id']).count()
    game_rate_count_sorted = game_rate_count.sort_values('id', ascending=False).index[:game_count].to_list()

    rows, col, rating = [], [], []

    for row in data.itertuples():
        game_id = getattr(row, 'game_id')
        if game_id in game_rate_count_sorted:        
            rows.append(users.index(getattr(row, 'user_id')))
            col.append(game_rate_count_sorted.index(game_id))
            rating.append(getattr(row, 'rating'))
    return csr_matrix((rating, (rows, col))), (users, game_rate_count_sorted)


# def make_dataframe(game_count, user_rate_limit, con):
#     query = 'SELECT * FROM boardgamers.recommend_review;'
#     data = pd.read_sql(query, con)

#     user_rate_count = data.groupby(['user_id']).count().sort_values('id', ascending=False)
#     if user_rate_limit:
#         limit_index = user_rate_count.id.tolist().index(user_rate_limit)
#     else:
#         limit_index = user_rate_count.size
#     user_rate_count = user_rate_count.index[:limit_index]
#     most_rated = data.groupby(['game_id']).count().sort_values('id', ascending=False).index[:game_count]
#     most_rated = most_rated.to_list()
    
#     df = data.pivot_table(index='user_id', columns='game_id', values='rating')

#     # most_rated에 속한 game들로만 이루어진 DataFrame 생성
#     column_df = pd.DataFrame(index=df.index, columns=['temp'])
#     for game in most_rated:
#         column_df[game] = df[game]
#     column_df.drop('temp', inplace=True, axis=1)
    
#     # user_rate_count에 속한 user들로만 이루어진 DataFrame 생성
#     if user_rate_limit:
#         complete_df = pd.DataFrame(index=['temp'], columns=most_rated)
#         for user in user_rate_count:
#             complete_df.loc[user] = column_df.loc[user]
#         complete_df.drop('temp', inplace=True)
#     # user_rate_limit이 0이라면(작성 리뷰 수 제한이 없다면) 따로 DataFrame을 생성하지 않는다
#     else:
#         complete_df = column_df
#     return complete_df.fillna(0)


def update_main(k, learning_rate, iteration, save_size):
    '''
    parameters
    - k : latent factors 수
    - learning_rate
    - iteration : 학습 반복 횟수
    - save_size : 예상 평점 상위 몇 개 저장할지
    '''

    ## password 보안용
    # config = Config(RepositoryEnv('boardGameRec/algorithms/.env'))
    # SQL_PWD = config('MYSQL_PASSWORD')
    SQL_PWD = 'qweasd123*'

    # SQL 서버와 연결
    engine = create_engine(f'mysql://ssafy:{SQL_PWD}@j5a404.p.ssafy.io/boardgamers')
    conn_alchemy =  engine.connect()

    # ratings = make_dataframe(1000, 0, conn_alchemy)
    ratings, matrix = make_csr(1000, conn_alchemy)
    mf = MatirxFactorization(ratings, k=k, learning_rate=learning_rate, iteration=iteration, save_size=save_size, engine=engine, matrix=matrix)
    mf.matrix_factorization()


if __name__ == '__main__':
    update_main(k=9, learning_rate=0.005, iteration=20, save_size=20)
    