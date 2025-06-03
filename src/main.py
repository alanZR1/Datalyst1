import sys
from pathlib import Path
import flet as ft

project_root = Path(__file__).parent.parent  
sys.path.append(str(project_root))  
 

from interface.main_app import MainApp


def main(page: ft.Page):
    try:
        page.title = "Datalyst"
        page.window.width = 1200
        page.window.height = 800
        page.scroll = ft.ScrollMode.AUTO 
        page.window_icon = "extras/logo.png" 
              
        app = MainApp(page)
        page.add(app)
        
        
    except Exception as e:
        print(f"Error en la aplicación: {str(e)}")
        raise
        

if __name__ == "__main__":
    ft.app(target = main)
