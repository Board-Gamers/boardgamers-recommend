[uwsgi]
chdir = /home/ubuntu/app/boardgamers/recommend/
module = recommend.wsgi:application
home = /home/ubuntu/app/boardgamers/recommend/venv/

uid = recommend_deploy
gid = recommend_deploy

http = :8081

enable-threads = true
master = true
vacuum = true
pidfile = /tmp/mysite.pid
logto = /home/ubuntu/app/boardgamers/recommend/log/uwsgi/recommend/@(exec://date +%%Y-%%m-%%d).log
log-reopen = true