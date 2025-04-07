import flet as ft
from interface.Data_Clean import DataCleanWindow



class MainApp(ft.Column):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.data = None
        self.kmeans = None
        self.df = None
        
        # Centrar los botones
        self.controls = [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Selecciona un modo:", size = 20, weight = "bold"),
                        ft.ElevatedButton("Online", on_click = self.open_online_window),
                        ft.ElevatedButton("procesamiento de Datos", 
                                          on_click = self.open_offline_window_clean_data),
                    ],
                    alignment = ft.MainAxisAlignment.CENTER,  
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER, 
                ),
                alignment=ft.alignment.center, 
                expand=True,  
            )
        ]


    def open_online_window(self, e):
        self.page.snack_bar = ft.SnackBar(ft.Text("Modo Online aún no implementado"), bgcolor="red")
        self.page.snack_bar.open = True
        self.page.update()
        
    # Limpia la ventana principal y abre la ventana secundaria
    # para el entranamiento
    def open_offline_window_clean_data(self, e):
        self.page.clean()
        DataCleanWindow(self.page)
 