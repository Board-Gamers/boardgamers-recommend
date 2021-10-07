import numpy as np
import pandas as pd
from decouple import Config, RepositoryEnv
from sqlalchemy import create_engine, text
from scipy.sparse import csr_matrix
import pymysql
pymysql.install_as_MySQLdb()


class MatirxFactorization:

    def __init__(self, game_count, k, learning_rate, iteration, save_size):
        # password 보안용
        config = Config(RepositoryEnv('boardGameRec/algorithms/.env'))
        SQL_PWD = config('MYSQL_PASSWORD')

        # SQL 서버와 연결
        self.engine = create_engine(f'mysql://ssafy:{SQL_PWD}@j5a404.p.ssafy.io/boardgamers')
        self.con = self.engine.connect()
        self.csr, self.users, self.games = self.make_csr(game_count)
        self.k = k
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.save_size = save_size


    '''
    일부 데이터만 활용하는 DataFrame 제작
    game_count : 평점 수를 기준으로 상위 x개만 알고리즘 실행
    '''
    def make_csr(self, game_count):
        self.con.execute(text('DELETE FROM boardgamers.recommend_review WHERE user_id >= 1000000;'))
        self.con.execute(text('INSERT INTO boardgamers.recommend_review (user_id, game_id, rating, from_bg) SELECT user_id + 1000000, game_id, rating, true FROM boardgamers.review'))
        query = 'SELECT * FROM boardgamers.recommend_review;'
        data = pd.read_sql(query, self.con)
        
        users = data.groupby(['user_id']).count().index.to_list()
        
        game_rate_count = data.groupby(['game_id']).count()
        game_rate_count_sorted = game_rate_count.sort_values('id', ascending=False).index[:game_count].to_list()

        rows, col, rating = [], [], []
        current_user = -2
        for row in data.itertuples():
            if getattr(row, 'user_id') != current_user:
                current_user = getattr(row, 'user_id')
                current_user_games = []
            game_id = getattr(row, 'game_id')
            if game_id in game_rate_count_sorted:
                # 한 유저가 같은 게임에 평점을 중복으로 등록한 경우 최근 것으로 덮어 씌운다
                if game_id in current_user_games:
                    idx_from_back = current_user_games.index(game_id) - len(current_user_games)
                    rating[idx_from_back] = getattr(row, 'rating')
                else:
                    rows.append(users.index(getattr(row, 'user_id')))
                    col.append(game_rate_count_sorted.index(game_id))
                    rating.append(getattr(row, 'rating'))
                    current_user_games.append(game_id)
        return csr_matrix((rating, (rows, col))), users, game_rate_count_sorted


    def matrix_factorization(self):
        self.gradient_descent()


    def gradient_descent(self):
        user_count, item_count = len(self.users), len(self.games)
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

                    error = prediction(P[i, :], Q[games[j], :], bu[i], bi[games[j]]) - r

                    delta_P, delta_bu = gradient(self.learning_rate, error, Q[games[j], :])
                    delta_Q, delta_bi = gradient(self.learning_rate, error, P[i, :])

                    P[i, :] -= delta_P
                    bu[i] -= delta_bu

                    Q[games[j], :] -= delta_Q
                    bi[games[j]] -= delta_bi

            print(iter, self.cost(P, Q, bu, bi))

        self.result_to_sql(P, Q, bu, bi)
        self.game_datas_to_sql(Q, bi)


    def cost(self, P, Q, bu, bi):
        cost = 0

        csr = self.csr
        ratings = csr.data
        games = csr.indices
        for i in range(csr.indptr.size-1):
            for j in range(csr.indptr[i], csr.indptr[i+1]):
                cost += pow(ratings[j] - prediction(P[i, :], Q[games[j], :], bu[i], bi[games[j]]), 2)
        return np.sqrt(cost/ratings.size)


    def result_to_sql(self, P, Q, bu, bi):
        result_to_save = []

        # BoardGameGeek 데이터가 아닌 실제 유저 id는 1000000부터 시작하므로 그 인덱스를 찾는다
        user_start = 1000000
        while True:
            try:
                user_index = self.users.index(user_start)
                break
            except:
                user_start += 1

        for i in range(user_index, len(self.users)):
            user_result = prediction(P[i, :], Q, bu[i], bi)
            user_id = self.users[i]
            
            games_rated = []
            for j in range(self.csr.indptr[i], self.csr.indptr[i+1]):
                games_rated.append(self.games[self.csr.indices[j]])

            user_result = pd.Series(user_result, self.games).sort_values(ascending=False)
            
            game_ids = user_result.keys()
            ratings = user_result.values

            saved = 0
            game_ids_idx = 0
            while saved < self.save_size:
                save_game_id = game_ids[game_ids_idx]
                if save_game_id not in games_rated:
                    result_to_save.append([user_id, game_ids[game_ids_idx], saved+1, ratings[game_ids_idx]])
                    saved += 1
                game_ids_idx += 1

        df_to_save = pd.DataFrame(result_to_save, columns=['user_id', 'game_id', 'rank', 'predicted_rating'])

        self.con.execute(text('TRUNCATE boardgamers.recommend_result'))
        df_to_save.to_sql(
            name='recommend_result',
            con=self.engine,
            if_exists='append',
            index=False
        )


    def game_datas_to_sql(self, Q, bi):
        weights_to_save = []
        for i in range(len(self.games)):
            for j in range(self.k):
                weights_to_save.append([self.games[i], j, Q[i][j]])

        df_to_save = pd.DataFrame(weights_to_save, columns=['game_id', 'k_num', 'weight'])

        self.con.execute(text('TRUNCATE boardgamers.recommend_game_latent_factor'))
        df_to_save.to_sql(
            name='recommend_game_latent_factor',
            con=self.engine,
            if_exists='append',
            index=False
        )

        bias_to_save = []
        for i in range(len(self.games)):
            bias_to_save.append([self.games[i], bi[i]])

        df_bias_to_save = pd.DataFrame(bias_to_save, columns=['game_id', 'bias'])

        self.con.execute(text('TRUNCATE boardgamers.recommend_game_bias'))
        df_bias_to_save.to_sql(
            name='recommend_game_bias',
            con=self.engine,
            if_exists='append',
            index=False
        )


def prediction(P, Q, bu, bi):
    return P.dot(Q.T) + bu + bi


def gradient(learning_rate, error, weight):
    weight_delta = learning_rate * np.dot(weight.T, error)
    bias_delta = learning_rate * np.sum(error)
    return weight_delta, bias_delta


def update_main(game_count, k, learning_rate, iteration, save_size):
    '''
    parameters
    - k : latent factors 수
    - learning_rate
    - iteration : 학습 반복 횟수
    - save_size : 예상 평점 상위 몇 개 저장할지
    '''
    mf = MatirxFactorization(game_count, k, learning_rate, iteration, save_size)
    mf.matrix_factorization()


if __name__ == '__main__':
    update_main(game_count=2000, k=15, learning_rate=0.005, iteration=5000, save_size=20)
    