import flet as ft
from interface.Data_Clean import DataCleanWindow



class MainApp(ft.Column):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.data = None
        self.kmeans = None
        self.df = None
        
        # Logo como imagen (ajusta la ruta)
        self.logo = ft.Image(
            src="extras/logo.png",
            width=150,
            height=150,
            fit=ft.ImageFit.CONTAIN
        )

        self.subtitle = ft.Text(
            "Aplicación de análisis y agrupamiento inteligente de datos",
            size=14,
            color=ft.Colors.GREY_600,
            italic=True,
            text_align=ft.TextAlign.CENTER,
            weight=ft.FontWeight.W_400
        )
        
        # Centrar los botones
        self.controls = [
            ft.Container(
                content=ft.Column(
                    [   
                        ft.Column([
                            self.logo,
                            self.subtitle,
                            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),  
                            ft.Divider(color=ft.Colors.GREY_300, thickness=1),
                            ],
                        spacing=5,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        
                        ft.Text("Selecciona un modo:", size = 20, weight = "bold"),
                        ft.ElevatedButton(
                                        "Online",
                                        icon = "Cloud",
                                        tooltip = "Ingresando a modo Online",
                                        on_click = self.open_online_window,
                                        style = ft.ButtonStyle(
                                            padding = 20
                                            ),
                                        ),
                        
                        ft.ElevatedButton(
                                        "procesamiento de Datos", 
                                        icon = "Data_Array",
                                        tooltip = "Ingresando a modo Offline",
                                        on_click = self.open_offline_window_clean_data,
                                        style = ft.ButtonStyle(
                                            padding = 20
                                            ),
                                        ),
                    ],
                    alignment = ft.MainAxisAlignment.CENTER,  
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER, 
                    spacing=20,
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
        
 