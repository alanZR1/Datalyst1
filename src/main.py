import asyncio
import flet as ft
from interface import main_app

""" crea una imagen de 800 x 600 de reso,
la instancua hacuia mainAPP en 
el script interface donde estan todos los parametros 
y controles 
"""

def main(page: ft.Page):
    
    try:
        page.title = "Datalyst"
        page.window.width = 800
        page.window.height = 600
        page.scroll = ft.ScrollMode.AUTO       
        app = main_app(page)
        page.add(app)
    except Exception as e:
        print(f"Error en la aplicación: {str(e)}")
        raise
        

if __name__ == "__main__":
    # Ejecuta la aplicación Flet
    ft.app(target = main)
