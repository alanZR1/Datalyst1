import flet as ft
from interface import MainApp

def main(page: ft.Page):
    page.title = "Modo de Selección"
    page.window.width = 800
    page.window_height = 600
    page.scroll = ft.ScrollMode.AUTO
    app = MainApp(page)
    page.add(app)

ft.app(target = main)
