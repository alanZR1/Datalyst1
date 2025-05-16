import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Apunta a la raíz del proyecto
import flet as ft
from interface.main_app import MainApp

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
    ft.app(target = main)
