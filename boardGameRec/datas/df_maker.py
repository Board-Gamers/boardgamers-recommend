import re
import sys
import xmltodict
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen


from pprint import pprint


def make_dataframe():
    game_ids = fetch_game_ids()
    games = []

    # API에 데이터가 없을 수도 있는 column들
    columns_check = {
        'thumbnail': 'thumbnail',
        'image': 'image',
        'description': 'description',
        'year_published': ['int', 'yearpublished'],
        'min_players': ['int', 'maxplayers'],
        'max_players': ['int', 'maxplayers'],
        'min_play_time': ['int', 'minplaytime'],
        'max_play_time': ['int', 'maxplaytime'],
        'min_age': ['int', 'age'],
        'category': ['list', 'boardgamecategory'],
        'play_type': ['list', 'boardgamemechanic'],
        'series': ['list', 'boardgameimplementation'],
        'designer': ['list', 'boardgamedesigner'],
        'artist': ['list', 'boardgameartist'],
        'publisher': ['list', 'boardgamepublisher']
    }

    for id in game_ids:
        url = f'https://www.boardgamegeek.com/xmlapi/boardgame/{id}'
        try:
            u = urlopen(url).read()
            doc = xmltodict.parse(u)
            game = doc['boardgames']['boardgame']

            data = {
                'id': id,
                'thumbnail': '',
                'image': '',
                'name': fetch_name(game['name']),
                'nameKor': fetch_korean_name(game['name']),
                'description': '',
                'year_published': '',
                'min_players': '',
                'max_players': '',
                'min_play_time': '',
                'max_play_time': '',
                'min_age': '',
                'category': '',
                'play_type': '',
                'series': '',
                'designer': '',
                'artist': '',
                'publisher': ''
            }

            # try 문을 사용, api에 해당 항목이 있다면 data 추가
            for key, val in columns_check.items():
                try:
                    if type(val) is str:
                        data[key] = game[val]
                    elif type(val) is list:
                        if val[0] == 'int':
                            data[key] = int(game[val[1]])
                        else:
                            data[key] = convert_to_list(game[val[1]])
                except:
                    pass

            games.append(data)
        except:
            pass

    columns = ['id', 'thumbnail', 'image', 'name', 'nameKor', 'description', 'year_published', 'min_players', 'max_players',
               'min_play_time', 'max_play_time', 'min_age', 'category', 'play_type', 'series', 'designer', 'artist', 'publisher']
    df = pd.DataFrame(games, columns=columns)

    df.to_csv('data.csv', sep=',', na_rep='', encoding='utf-8-sig')


def fetch_game_ids():
    sys.stdin = open('game_ids.txt', 'r')

    rows = []
    for _ in range(19229):
        x = int(input())
        rows.append(x)
    return rows


def fetch_name(names):
    if type(names) is list:
        return names[0]['#text']
    return names['#text']


def fetch_korean_name(names):
    try:
        for i in names[1:]:
            name = i['#text']
            if isHangul(name):
                return name
    except:
        pass

    url = "http://boardlife.co.kr/game_rank.php?search="
    name_eng = fetch_name(names).replace(' ', '+')
    url += name_eng

    selector = 'body > table tr:nth-child(2) > td > table tr:nth-child(1) > .ellip > a'

    b_soup = BeautifulSoup(requests.get(url).text, "html.parser")
    try:
        name = b_soup.select(selector)[0].text
        return name.strip()
    except:
        pass


def convert_to_list(dict_list):
    result = []
    if type(dict_list) is list:
        for category in dict_list:
            result.append(category['#text'])
    else:
        result.append(dict_list['#text'])
    return str(result)


def isHangul(text):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', text))
    return hanCount > 0


if __name__ == '__main__':
    make_dataframe()
