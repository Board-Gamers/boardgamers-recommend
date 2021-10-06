from django.http import HttpResponse
from .algorithms import matrix_factorization
from django.views.decorators.http import require_POST


# Create your views here.
def test(request):
    response = 'Program is working well.'
    return HttpResponse(response)
    

def update_gd(request):
    matrix_factorization.update_main(9, 0.005, 300, 20)
    return HttpResponse(status=200)
