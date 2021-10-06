import schedule
import time
from matrix_factorization import update_main


def job():
    update_main(game_count=2000, k=12, learning_rate=0.005, iteration=300, save_size=20)


if __name__ == '__main__':
    schedule.every().day.at("01:00").do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(5)
