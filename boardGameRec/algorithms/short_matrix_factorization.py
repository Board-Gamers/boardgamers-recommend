import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import pymysql
pymysql.install_as_MySQLdb()
from .matrix_factorization import prediction, gradient


class ShortMatrixFactorization:

    def __init__(self, user_id, learning_rate, iteration, save_size):
        # password 보안용
        config = Config(RepositoryEnv('boardGameRec/algorithms/.env'))
        SQL_PWD = config('MYSQL_PASSWORD')

        self.user_id = user_id + 1000000
        self.engine = create_engine(f'mysql://ssafy:{SQL_PWD}@j5a404.p.ssafy.io/boardgamers')
        self.con = self.engine.connect()
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.save_size = save_size


    def short_gradient_descent(self):
        self.con.execute(text('DELETE FROM boardgamers.recommend_review WHERE user_id >= 1000000;'))
        self.con.execute(text('INSERT INTO boardgamers.recommend_review (user_id, game_id, rating, from_bg) SELECT user_id + 1000000, game_id, rating, true FROM boardgamers.review'))

        Q_query = 'SELECT * FROM boardgamers.recommend_game_latent_factor;'
        Q_data = pd.read_sql(Q_query, self.con)
        Q = Q_data.pivot_table(index='game_id', columns='k_num', values='weight')
        k = Q.columns.size
        bi_query = 'SELECT * FROM boardgamers.recommend_game_bias;'
        bi_data = pd.read_sql(bi_query, self.con)
        bi = bi_data.pivot_table(index='game_id', values='bias').loc[:, 'bias']
        
        P = np.random.normal(size=(k))
        bu = np.zeros(1)

        query = f'SELECT * FROM boardgamers.recommend_review WHERE user_id = {self.user_id};'
        data = pd.read_sql(query, self.con)
        user_games = []
        # user가 평점을 매긴 game 중 상위 game_count에 드는 게임만 추려낸다
        for g in data.loc[:, 'game_id'].to_numpy():
            if g in Q.index:
                user_games.append(g)

        fil = data.game_id.apply(lambda x: x in user_games)
        data = data[fil]
        for iter in range(self.iteration):
            for row in data.itertuples():
                game_id = getattr(row, 'game_id')

                r = getattr(row, 'rating')
                error = prediction(P[:], Q.iloc[Q.index.to_list().index(game_id), :], bu[0], bi.loc[game_id]) - r
                delta_P, delta_bu = gradient(self.learning_rate, error, Q.iloc[Q.index.to_list().index(game_id)])
                P -= delta_P
                bu[0] -= delta_bu

            print(iter, self.cost(P, Q, bu, bi, data))

        self.result_to_sql(P, Q, bu, bi, data)


    def cost(self, P, Q, bu, bi, data):
        cost = 0

        pred = prediction(P, Q, bu, bi)
        for row in data.itertuples():
            game_id = getattr(row, 'game_id')

            cost += pow(getattr(row, 'rating') - pred.loc[game_id], 2)
        return np.sqrt(cost/data.size)


    def result_to_sql(self, P, Q, bu, bi, data):
        result_to_save = []

        pred = prediction(P, Q, bu, bi).sort_values(ascending=False)
        game_ids = pred.index

        saved = 0
        game_ids_idx =0
        while saved < self.save_size:
            save_game_id = game_ids[game_ids_idx]

            if save_game_id not in data.loc[:, 'game_id']:
                result_to_save.append([self.user_id, save_game_id, saved+1, pred.iloc[game_ids_idx]])
                saved += 1
            game_ids_idx += 1
        
        df_to_save = pd.DataFrame(result_to_save, columns=['user_id', 'game_id', 'rank', 'predicted_rating'])

        self.con.execute(text(f'DELETE FROM boardgamers.recommend_result WHERE user_id = {self.user_id}'))
        df_to_save.to_sql(
            name='recommend_result',
            con=self.engine,
            if_exists='append',
            index=False
        )


def update_one_user(user_id, learning_rate, iteration, save_size):
    smf = ShortMatrixFactorization(user_id, learning_rate, iteration, save_size)
    smf.short_gradient_descent()


if __name__ == '__main__':
    update_one_user(user_id=7, learning_rate=0.005, iteration=1000, save_size=20)
