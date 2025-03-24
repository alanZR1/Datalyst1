import flet as ft
from interface import MainApp

""" crea una imagen de 800 x 600 de reso,
la instancua hacuia mainAPP en 
el script interface donde estan todos los parametros 
y controles 
"""

def main(page: ft.Page):
    page.title = "Datalyst"
    page.window.width = 800
    page.window_height = 600
    page.scroll = ft.ScrollMode.AUTO 
    app = MainApp(page)
    page.add(app)

ft.app(target = main)
