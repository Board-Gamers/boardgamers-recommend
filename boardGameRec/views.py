from django.http import HttpResponse
from .algorithms import matrix_factorization, short_matrix_factorization
from django.views.decorators.http import require_POST


# Create your views here.
def test(request):
    response = 'Program is working well.'
    return HttpResponse(response)


@require_POST
def test_post(request):
    response = 'Program is working well.'
    return HttpResponse(response)
    

@require_POST
def update_gd(request):
    matrix_factorization.update_main(9, 0.005, 300, 20)
    response = '추천 결과가 업데이트 되었습니다.'
    return HttpResponse(response, status=200)


def update_gd_one(request, user_id):
    short_matrix_factorization.update_one_user(user_id, 0.005, 3000, 20)
    response = '실시간 추천 결과가 업데이트 되었습니다.'
    return HttpResponse(response, status=200)
