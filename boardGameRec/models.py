from django.db import models

# Create your models here.


class Game(models.Model):
    id = models.IntegerField(primary_key=True)
    thumbnail = models.CharField(max_length=1000)
    image = models.CharField(max_length=1000)
    name = models.CharField(max_length=100)  # primary
    nameKor = models.CharField(max_length=100)  # 리스트
    description = models.TextField()
    yearPublished = models.IntegerField()
    minPlayers = models.IntegerField()
    maxPlayers = models.IntegerField()
    minPlayTime = models.IntegerField()
    maxPlayTime = models.IntegerField()
    minAge = models.IntegerField()
    category = models.TextField()  # 리스트
    playType = models.TextField()  # 리스트 # boardgamemechanic
    series = models.TextField()  # 리스트 # boardgameimplementation
    designer = models.CharField(max_length=100)  # boardgamedesigner
    artist = models.TextField()  # 리스트 # boardgameartist
    publisher = models.TextField()  # 리스트 # boardgamepublisher
    usersRated = models.IntegerField()
    averageRate = models.FloatField()  # average
    rank = models.IntegerField()  # Board Game Rank


class Review(models.Model):
    user = models.CharField(max_length=50)
    rating = models.FloatField()
    comment = models.TextField()
    gameId = models.ForeignKey(Game, on_delete=models.CASCADE)
    gameName = models.CharField(max_length=100)
    createdAt = models.DateTimeField(auto_now_add=True)
    updatedAt = models.DateTimeField(auto_now=True)
