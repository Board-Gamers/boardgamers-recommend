import re
import sys
import xmltodict
import pandas as pd
from urllib.request import urlopen


from pprint import pprint


def make_dataframe():
    game_ids = fetch_game_ids()
    games = []
    for id in game_ids[:50]:
        url = f'https://www.boardgamegeek.com/xmlapi/boardgame/{id}'
        u = urlopen(url).read()
        doc = xmltodict.parse(u)
        game = doc['boardgames']['boardgame']
        
        games.append({
            'id': id,
            'thumbnail': game['thumbnail'],
            'image': game['image'],
            'name': fetch_name(game['name']),
            'nameKor': fetch_korean_name(game['name']),
            'description': game['description'],
            'yearPublished': int(game['yearpublished']),
            'minPlayers': int(game['minplayers']),
            'maxPlayers': int(game['maxplayers']),
            'minPlayTime': int(game['minplaytime']),
            'maxPlayTime': int(game['maxplaytime']),
            'minAge': int(game['minAge']),
        })


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


def isHangul(text):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', text))
    return hanCount > 0


if __name__ == '__main__':
    make_dataframe()
