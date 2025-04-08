import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Apunta a la raíz del proyecto
import flet as ft
from interface.main_app import MainApp

""" crea una imagen de 800 x 600 de resolucion,
la instancua hacia main_app en 
el script interface donde estan todos los parametros 
y controles"""

def main(page: ft.Page):
    
    try:
        page.title = "Datalyst"
        page.window.width = 800
        page.window.height = 600
        page.scroll = ft.ScrollMode.AUTO       
        app = MainApp(page)
        page.add(app)
    except Exception as e:
        print(f"Error en la aplicación: {str(e)}")
        raise
        

if __name__ == "__main__":
    # Ejecuta la aplicación Flet
    ft.app(target = main)
