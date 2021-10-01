from django.http import HttpResponse
from .algorithms import matrix_factorization
from django.views.decorators.http import require_POST


# Create your views here.
# @require_POST
def update_gd(request):
    matrix_factorization.update_main()
    return HttpResponse(status=200)
