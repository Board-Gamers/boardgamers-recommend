import re
import sys
import xmltodict
import pandas as pd
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
        'yearPublished': ['int', 'yearpublished'],
        'minPlayers': ['int', 'maxplayers'],
        'maxPlayers': ['int', 'maxplayers'],
        'minPlayTime': ['int', 'minplaytime'],
        'maxPlayTime': ['int', 'maxplaytime'],
        'minAge': ['int', 'age'],
        'category': ['list', 'boardgamecategory'],
        'playType': ['list', 'boardgamemechanic'],
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
                'yearPublished': '',
                'minPlayers': '',
                'maxPlayers': '',
                'minPlayTime': '',
                'maxPlayTime': '',
                'minAge': '',
                'category': '',
                'playType': '',
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

    columns = ['id', 'thumbnail', 'image', 'name', 'nameKor', 'description', 'yearPublished', 'minPlayers', 'maxPlayers',
               'minPlayTime', 'maxPlayTime', 'minAge', 'category', 'playType', 'series', 'designer', 'artist', 'publisher']
    df = pd.DataFrame(games, columns=columns)

    df.to_csv('data.csv', sep=',', na_rep='', encoding='utf-8-sig')


def fetch_game_ids():
    sys.stdin = open('game_ids.txt', 'r')

    rows = []
    for _ in range(19229):
        x = int(input())
        rows.append(x)
    return rows


def fetch_name(dict):
    if type(dict) is list:
        return dict[0]['#text']
    return dict['#text']


def fetch_korean_name(names):
    try:
        for i in names:
            name = i['#text']
            if isHangul(name):
                return name
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
