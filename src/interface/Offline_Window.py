import flet as ft
#import pandas as pd
from src.clustering.clustering import train_kmeans, calculate_silhouette
from src.models.model_save_carge import save_model    
#clase para la venta de entrenamiento

class OfflineWindow(ft.Column):
    def __init__(self, page, cleaned_df):
        print("Inicializando OfflineWindow")
        try:
            super().__init__(expand=True, scroll=ft.ScrollMode.AUTO)
            self.page = page
            self.df = cleaned_df
            self.kmeans = None

            # Controles
            self.k_input = ft.TextField(label="Número de clusters", value="2")
            self.n_init_input = ft.TextField(label="Número de iteraciones", value="10")

            # Configuración de dropdowns
            numeric_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
            dropdown_options = [ft.dropdown.Option(col) for col in numeric_columns]

            self.x_axis_dropdown = ft.Dropdown(
                label="Característica X",
                options=dropdown_options,
                value=numeric_columns[0] if numeric_columns else None,
                width=200,
                #height=60
            )

            self.y_axis_dropdown = ft.Dropdown(
                label="Característica Y",
                options=dropdown_options,
                value=numeric_columns[1] if len(numeric_columns) > 1 else None,
                width=200,
                #height=60
            )

            # Área de visualización
            self.image = ft.Image(
                width=500,
                height=400,
                src_base64="EN ESPERA...",
                border_radius=10,
                fit=ft.ImageFit.CONTAIN
            )

            self.train_button = ft.ElevatedButton(
                "Entrenar",
                on_click=self.train_kmeans,
                icon="play_arrow",
            )

            self.save_model_button = ft.ElevatedButton(
                "Guardar Modelo",
                on_click=self.save_model,
                visible=False,
                icon="save_alt",
            )

            self.silhouette_text = ft.Text(
                "Índice de silueta: -",
                size=14,
                weight="bold",
                color="black"
            )
            
            self.back_button = ft.FloatingActionButton(
                icon = "Arrow_Back",
                bgcolor= ft.Colors.BLUE_700,
                shape=ft.CircleBorder(),
                autofocus=True,
                tooltip="Volver",
                mini=True,
                on_click= self.go_back
            )

            # Layout
            self.controls = [
                ft.Stack(
                [
                    ft.Row(
                        [
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Text("Entrenamiento de K-Means", size=18,    weight="bold"),
                                        #ft.Text("Datos ya procesados", color=ft.colors.    GREEN),
                                        ft.Divider(height=1),
                                        self.k_input,
                                        self.n_init_input,
                                        ft.Divider(),
                                        self.x_axis_dropdown,
                                        self.y_axis_dropdown,
                                        ft.Row(
                                            [
                                                self.train_button, 
                                                self.save_model_button
                                            ],
                                            spacing=10,
                                            ),
                                        self.silhouette_text
                                    ],
                                    spacing=8,
                                ),
                                width=350,
                                padding=15
                            ),
                            ft.Container(
                                content=self.image,
                                expand=True,
                                padding=15
                            )
                        ],
                        expand=True,
                    ),
                    ft.Container(
                        content = self.back_button,
                        alignment=ft.alignment.top_right,
                        margin=ft.margin.only(right=10, bottom=10),
                    )
                ],
                expand=True,
                )
            ]
            #self.page.controls.clear()  # Limpia pantalla anterior
            #self.page.add(ft.Container(content=self, expand=True))  # Contenedor principal
            #self.page.update()
            #self.expand = True
        except Exception as ex:
            print(f"Error en OfflineWindow.__init__: {ex}")

    def update_preview(self, e):
        """Actualiza la vista previa cuando cambian las selecciones"""
        if not hasattr(self, 'image'):
            return

        try:
            x_col = self.x_axis_dropdown.value
            y_col = self.y_axis_dropdown.value

            if x_col and y_col:
                # Muestra datos de muestra en consola para debug
                print(f"Columnas seleccionadas: {x_col}, {y_col}")
                print(self.df[[x_col, y_col]].head())
# Muestra datos de muestra en consola para debug
#print(f"Columnas seleccionadas: {x_col}, {y_col}")
#print(self.df[[x_col, y_col]].head())

        except Exception as ex:
            print(f"Error al actualizar vista previa: {ex}")


    def train_kmeans(self, e):
        try:
            k = int(self.k_input.value)
            n_init = int(self.n_init_input.value)
            x_col = self.x_axis_dropdown.value
            y_col = self.y_axis_dropdown.value

            if not all ([x_col, y_col]):
                raise ValueError("Selecciona ambas columnas para el entrenamiento.")

            img_base64, self.kmeans = train_kmeans(self.df, k, n_init, x_col, y_col)

            self.image.src_base64 = img_base64
            self.save_model_button.visible = True


            score = calculate_silhouette(self.df, self.kmeans)
            self.update_silhouette_score(score)

            self.page.update()
        except Exception as ex:
            self.show_snackbar(f"Error en entrenamiento: {str(ex)}", "red")

    def update_silhouette_score(self, score: float = None):
        """Actualiza el texto del índice de silueta con formato condicional"""
        if score is not None:
            # Formateo y coloreado basado en el valor
            self.silhouette_text.value = f"Índice de silueta: {score:.4f}"

            # Asigna color según calidad del clustering
            if score > 0.5:
                self.silhouette_text.color = "green"
            elif score > 0.25:
                self.silhouette_text.color = "orange"
            else:
                self.silhouette_text.color = "red"
        else:
            # Estado por defecto/reseteo
            self.silhouette_text.value = "Índice de silueta: -"
            self.silhouette_text.color = ft.colors.BLACK

        self.page.update()
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
            
    def go_back(self, e):
        self.page.overlay.clear()
        from src.interface.Data_Clean import DataCleanWindow
        self.page.clean()
        self.page.add(DataCleanWindow(self.page))
        self.page.update()

    def show_snackbar(self, message: str, color: str = "green"):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        self.page.snack_bar.open = True
        self.page.update()