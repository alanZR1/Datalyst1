import flet as ft
from src.clustering.clustering import train_kmeans, calculate_silhouette, calculate_optimal_k
from src.models.model_save_carge import save_model
#clase para la venta de entrenamiento
        
class OfflineWindow(ft.Column):
    def __init__(self, page, cleaned_df):
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
            height=60
        )

        self.y_axis_dropdown = ft.Dropdown(
            label="Característica Y",
            options=dropdown_options,
            value=numeric_columns[1] if len(numeric_columns) > 1 else None,
            width=200,
            height=60
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
            icon=ft.icons.PLAY_ARROW
        )

        self.save_model_button = ft.ElevatedButton(
            "Guardar Modelo",
            on_click=self.save_model,
            visible=False,
            icon=ft.icons.SAVE
        )
        
        self.silhouette_text = ft.Text(
            "Índice de silueta: -",
            size=14,
            weight="bold",
            color=ft.colors.BLACK
        )
        
        self.calculate_k_btn = ft.ElevatedButton(
            "calcular k optimo",
            on_click = self.calculate_optimal_k,
            icon=ft.icons.CALCULATE,
            tooltip="Calcula el número óptimo de clusters"    
        )
        
        self.k_method = ft.Dropdown(
            label = "metodo de cálculo",
            options=[
                ft.dropdown.Option("(Jambú), método de Jambú"),
                ft.dropdown.Option("(Silhouette), método de silueta"),
            ],
            width=200,
            value="Silueta"
        )
        
        self.k_resutl = ft.Text(
            "Resultado de k óptimo: --",
            size=14,
            color=ft.colors.BLUE
            ) 
        # Layout
        self.controls = [
            ft.Row(
                [
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("Entrenamiento de K-Means", size=18, weight="bold"),
                                #ft.Text("Datos ya procesados", color=ft.colors.GREEN),
                                ft.Divider(height=1),
                                ft.text("calculo de k óptimo", weight="bold"),
                                ft.row([self.k_method, self.calculate_k_btn]),
                                self.k_resutl,
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
            )
        ]

    def update_preview(self, e):
        if not hasattr(self, 'image'):
            return
        
        try:
            x_col = self.x_axis_dropdown.value
            y_col = self.y_axis_dropdown.value
        
            #if x_col and y_col:
                # Muestra datos de muestra en consola para debug
                #print(f"Columnas seleccionadas: {x_col}, {y_col}")
                #print(self.df[[x_col, y_col]].head())
            
        except Exception as ex:
            print(f"Error al actualizar vista previa: {ex}")

    def calculate_optimal_k(self, e):
        try:
            x_col = self.x_axis_dropdown.value
            y_col = self.y_axis_dropdown.value
            
            if not all([x_col, y_col]):
                raise ValueError("Selecciona ambas columnas")
            
            data = self.df[[x_col, y_col]].values
            method = self.k_method.value  # "jambu", "silhouette" o "elbow"
            
            # Llama a la función centralizada
            optimal_k = calculate_optimal_k(data, method=method)
            
            self.k_input.value = str(optimal_k)
            self.k_result.value = f"K óptimo: {optimal_k}"
            
        except Exception as ex:
            self.show_snackbar(f"Error calculando K: {str(ex)}", "red")
        finally:
            self.page.splash = None
            self.page.update()
                
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
                self.silhouette_text.color = ft.colors.GREEN
            elif score > 0.25:
                self.silhouette_text.color = ft.colors.ORANGE
            else:
                self.silhouette_text.color = ft.colors.RED
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

    def show_snackbar(self, message: str, color: str = "green"):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        self.page.snack_bar.open = True
        self.page.update()
        
