from django.db import models

# Create your models here.


class Game(models.Model):
    id = models.IntegerField(primary_key=True)
    thumbnail = models.CharField(max_length=1000)
    image = models.CharField(max_length=1000)
    name = models.CharField(max_length=100)  # primary
    name_kor = models.CharField(max_length=100)  # 리스트
    description = models.TextField()
    year_published = models.IntegerField()
    min_players = models.IntegerField()
    max_players = models.IntegerField()
    min_play_time = models.IntegerField()
    max_play_time = models.IntegerField()
    min_age = models.IntegerField()
    category = models.TextField()  # 리스트
    play_type = models.TextField()  # 리스트 # boardgamemechanic
    series = models.TextField()  # 리스트 # boardgameimplementation
    designer = models.TextField() # 리스트  # boardgamedesigner
    artist = models.TextField()  # 리스트 # boardgameartist
    publisher = models.TextField()  # 리스트 # boardgamepublisher
    users_rated = models.IntegerField()
    average_rate = models.FloatField()  # average
    rank = models.IntegerField()  # Board Game Rank


class Review(models.Model):
    user = models.CharField(max_length=50)
    rating = models.FloatField()
    comment = models.TextField()
    game = models.ForeignKey(Game, on_delete=models.CASCADE)
    game_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
