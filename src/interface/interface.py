import flet as ft
from data_processing import load_csv, clean_data
from clustering import train_kmeans, calculate_silhouette
from model_save_carge import save_model, load_model
import traceback
import pandas  as pd

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
        
        
class DataCleanWindow(ft.Column):
    print("se ingreso a procesamiento de datos")
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.df = None
        self.cleaning_params = {
            "remove_outliers": False,
            "fill_na": "none", # "mean" | "median" | "mode"
            "remove_duplicates": False,
            "normalize": False
        }
         
        self.continue_btn = ft.ElevatedButton(
            "Continuar a Entrenamiento",
            on_click = self.apply_cleaning,
            disabled = True
        )
        
        # Dropdowns SOLO para vista previa (sin funcionalidad de entrenamiento)
        self.preview_x_dropdown = ft.Dropdown(
            label="Vista previa - Eje X", 
            options=[], 
            disabled=True
        )
        self.preview_y_dropdown = ft.Dropdown(
            label="Vista previa - Eje Y", 
            options=[], 
            disabled=True
        )
                # Botón de actualizar vista previa
        self.update_preview_btn = ft.ElevatedButton(
            "Actualizar Vista Previa",
            on_click=self.update_preview,
            disabled=True
        )
        
        self.file_picker = ft.FilePicker(on_result = self.file_selected)
        self.page.overlay.append(self.file_picker)

        self.remove_outliers = ft.Switch(label="Eliminar Outliers", value=False)
        self.fill_na = ft.Dropdown(
            label = "Rellenar valores nulos",
            options = [
                ft.dropdown.Option("none", "No rellenar"),
                ft.dropdown.Option("mean", "Media"),
                ft.dropdown.Option("median", "Mediana"),
            ],
            value="none"
        )
        
        self.normalize_data = ft.Switch(label = "Normalizar datos", value = False)
        self.remove_duplicates = ft.Switch(label = "Eliminar duplicados", value = False)
        #vista previa de esos datos
        self.data_preview = ft.DataTable(
            columns = [ft.DataColumn(ft.Text("Cargando datos..."))],
            rows = []
        )
        # diseño de la ventana secundaria
        self.controls = [
            ft.Row([
                    ft.Column([
                            ft.Text("Preprocesamiento de Datos", size=18, weight="bold"),
                            ft.ElevatedButton("Cargar CSV", on_click=self.pick_file),
                            ft.Text("Opciones de limpieza:", size=16),
                            self.remove_outliers,
                            self.fill_na,
                            self.normalize_data,
                            self.remove_duplicates,
                            self.continue_btn,
                            self.preview_x_dropdown,
                            self.preview_y_dropdown,
                            self.update_preview_btn
                        ],
                        width = 400
                    ),
                    ft.Container(
                        content = ft.Column([
                            ft.Text("Vista previa de datos:", size = 16),
                            ft.Container(
                                content = self.data_preview,
                                border = ft.border.all(1, ft.colors.GREY_400),
                                padding = 10,
                                border_radius = 10,
                            )
                        ]),
                    expand = True
                    )
                ],
                expand = True
            )
        ]
        self.page.add(self)
        self.page.update()

    def pick_file(self, e):
        self.file_picker.pick_files(allow_multiple = False)

    def file_selected(self, e: ft.FilePickerResultEvent):
        if not e.files:
            return

        try:
            file_path = e.files[0].path
            print(f"\n=== Debug inicio ===")
            print(f"Ruta del archivo: {file_path}")

            # 1. Carga el archivo
            self.df = load_csv(file_path)  # Carga el archivo CSV

           
                        # Actualiza la vista previa
            numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            self.preview_x_dropdown.options = [ft.dropdown.Option(col) for col in numeric_cols]
            self.preview_y_dropdown.options = [ft.dropdown.Option(col) for col in numeric_cols]
            
            #habilita los dropdowns
            self.preview_x_dropdown.disabled = False
            self.preview_y_dropdown.disabled = False
            self.update_preview_btn.disabled = False
            
            self.controls[0].controls[0].controls[-1].disabled = False
             # 3. Asignación solo si todo está bien
            print(f"Columnas válidas: {self.df.columns.tolist()}")
            
            self.show_snackbar("Archivo cargado correctamente", "green")
            # 4. Actualización UI
            self.update_preview()
            self.enable_training_button()

            print("=== Debug fin ===\n")

        except Exception as ex:
            error_msg = f"Error al cargar {file_path}: {str(ex)}"
            print(f"\n!!! ERROR: {error_msg}")
            print(traceback.format_exc())
            self.show_snackbar(error_msg, "red")
            self.df = None  # Asegura resetear el DataFrame 
            
    def enable_training_button(self):
        # Habilita el botón de entrenamiento si hay datos cargados
        if hasattr (self.df, 'columns') and not self.df.empty:
            self.continue_btn.disabled = False
            self.controls[0].controls[0].controls[-1].disabled = False
            self.page.update()
            
    def update_preview(self):
        try:
            if self.df is None:
                raise ValueError("No hay datos cargados")
            x_col = self.preview_x_dropdown.value
            y_col = self.preview_y_dropdown.value

            if not x_col or not y_col:
                raise ValueError("Selecciona ambas columnas para la vista previa.")
            self.update_data_table()
            self.show_snackbar("Vista previa actualizada", "green")
            # Actualiza la vista previa
            numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            self.preview_x_dropdown.options = [ft.dropdown.Option(col) for col in numeric_cols]
            self.preview_y_dropdown.options = [ft.dropdown.Option(col) for col in numeric_cols]
            
            #habilita los dropdowns
            self.preview_x_dropdown.disabled = False
            self.preview_y_dropdown.disabled = False
            self.update_preview_btn.disabled = False

            # Actualiza DataTable
            self.update_data_table()
            self.show_snackbar("Vista previa actualizada", "green")

        except Exception as ex:
            print(f"Error en update_preview: {traceback.format_exc()}")
            self.show_snackbar(f"Error al mostrar vista previa: {str(ex)}", "red")
            
    def update_data_table(self):
        """Actualiza la tabla de vista previa con los datos actuales"""
        try:
            if self.df is None or self.df.empty:
                return

            # Crear columnas para el DataTable
            columns = [ft.DataColumn(ft.Text(col)) for col in self.df.columns[:5]]  # Mostrar máximo 5 columnas
        
            # Crear filas con los primeros registros
            rows = []
            for _, row in self.df.head(5).iterrows():  # Mostrar máximo 5 filas
                cells = [ft.DataCell(ft.Text(str(row[col]))) for col in self.df.columns[:5]]
                rows.append(ft.DataRow(cells=cells))
        
            self.data_preview.columns = columns
            self.data_preview.rows = rows
            self.page.update()
        
        except Exception as ex:
            print(f"Error actualizando tabla: {str(ex)}")
            self.show_snackbar("Error al mostrar datos", "red")
            
    def apply_cleaning(self, e):
        """Aplica los parámetros de limpieza y pasa a la ventana de entrenamiento"""
        try:
            if self.df is None or self.df.empty:
                raise ValueError("No hay datos cargados para limpiar")

            # Mostrar indicador de procesamiento
            self.page.splash = ft.ProgressBar()
            self.page.update()

            # Aplicar limpieza con los parámetros actuales
            cleaned_df = clean_data(
                df=self.df,
                remove_outliers=self.remove_outliers.value,
                fill_na=self.fill_na.value,
                remove_duplicates=self.remove_duplicates.value,
                normalize=self.normalize_data.value
            )

            # Verificar que quedaron datos después de la limpieza
            if cleaned_df.empty:
                raise ValueError("El DataFrame quedó vacío después de la limpieza")

            # Pasar a ventana de entrenamiento
            self.page.clean()
            OfflineWindow(self.page, cleaned_df)

        except Exception as ex:
            self.show_snackbar(f"Error en limpieza: {str(ex)}", "red")
        finally:
            # Asegurarse de quitar el indicador de carga
            self.page.splash = None
            self.page.update()
    def show_snackbar(self, message: str, color: str = "green"):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor = color)
        self.page.snack_bar.open = True
        self.page.update()


        
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
        
