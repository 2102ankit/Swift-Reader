from django.contrib import admin

# Register your models here.
from .models import News

class NewsAdmin(admin.ModelAdmin):
    list_display=('headline','summary','contentURL','imageURL','category','created','id')


admin.site.register(News,NewsAdmin)