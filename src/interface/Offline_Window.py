import flet as ft
import pandas as pd
from src.clustering.clustering import train_kmeans, calculate_silhouette
 
from src.models.model_save_carge import save_model    
#clase para la venta de entrenamiento
        
class OfflineWindow(ft.Column):
    print("se ingreso a entrenamiento ")
    def __init__(self, page, cleaned_df):
        super().__init__()
        self.page = page
        self.df = cleaned_df #datos ya limpios
        self.kmeans = None

        #controles de la ventana secundaria
        self.k_input = ft.TextField(label="Número de clusters", value="0")
        self.n_init_input = ft.TextField(label="Número de iteraciones", value="0")
        
        #dropdowns para seleccionar las columnas
        numeric_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
        self.x_axis_dropdown = ft.Dropdown(
            label = "Característica X",
            options = [],
            value = numeric_columns[0] if numeric_columns else None
            )
        self.y_axis_dropdown = ft.Dropdown(
            label = "Característica Y",
            options = [],
            value = numeric_columns[1] if len(numeric_columns) > 1 else None
            )
        
        # area de visualizacion
        self.image = ft.Image(
            width = 500,
            height = 400,
            src_base64 = " EN ESPERA...",
            #bgcolor = ft.colors.GREY_300,
            border_radius = 10,
            fit = ft.ImageFit.CONTAIN  
        )
        self.image_container = ft.Container(
            content = self.image,
            bgcolor = ft.colors.GREY_300,
            padding = 10,
            border_radius = 10
        )
        
        # Botones y resultados
        
        self.train_button = ft.ElevatedButton(
            "Entrenar",
            on_click = self.train_kmeans,
            icon = ft.icons.PLAY_ARROW
        )
        self.silhouette_text = ft.Text("Índice de silueta: -")
        self.save_model_button = ft.ElevatedButton(
            "Guardar Modelo",
            on_click = self.save_model,
            visible = False,
            icon = ft.icons.SAVE
        )

        # Diseño de la ventana secundaria
        self.controls = [
            ft.Row(
                [
                    ft.Column(
                        [
                            ft.text.Text("Entrenamiento de K-Means", size = 18, weight = "bold"),
                            ft.text.Text("Datos ya procesados", color = ft.colors.GREEN),
                            ft.Divider(),
                            self.k_input,
                            self.n_init_input,
                            ft.text.Text("Seleccion de los ejes", weight = "bold"),
                            ft.Row([self.x_axis_dropdown, self.y_axis_dropdown]),
                            ft.Divider(),
                            self.train_button,
                            self.save_model_button,
                            ft.Divider(),
                            self.silhouette_text,
                        ],
                        spacing = 15,
                        width = 300
                    ),
                    self.image_container
                ],
                spacing = 20,
                expand = True
            )
        ]

        # Actualiza la página
        self.page.add(self)
        self.page.update()
        
    def train_kmeans(self, e):
        try:
            k = int(self.k_input.value)
            n_init = int(self.n_init_input.value)
            x_col = self.x_axis_dropdown.value
            y_col = self.y_axis_dropdown.value

            if not all ([x_col, y_col]):
                raise ValueError("Por favor selecciona ambas columnas para el entrenamiento.")
            
            img_base64, self.kmeans = train_kmeans(self.df, k, n_init, x_col, y_col)
            
            self.image.src_base64 = img_base64
            self.save_model_button.visible = True
            #calcula silueta automaticamente
            score = calculate_silhouette(self.df, self.kmeans)
            self.silhouette_text.value = f"Índice de silueta: {score:.4f}"
            
            self.page.update()
        except Exception as ex:
            self.show_snackbar(f"Error en entrenamiento: {str(ex)}", "red")

    def save_model(self, e):
        save_file_dialog = ft.FilePicker(on_result = self.on_save_model)
        self.page.overlay.append(save_file_dialog)
        self.page.update()
        save_file_dialog.save_file()

    def on_save_model(self, e: ft.FilePickerResultEvent):
        if e.path:
            try:
                save_model(self.kmeans, e.path)
                self.show_snackbar(f"✅ Modelo guardado en: {e.path}")
                self.page.update()
            except Exception as ex:
                self.show_snackbar(f"Error al guardar el modelo: {str(ex)}", "red")
            self.page.update()

    def show_snackbar(self, message: str, color: str = "green"):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        self.page.snack_bar.open = True
        self.page.update()
        
