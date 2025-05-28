import flet as ft
from src.interface.Offline_Window import OfflineWindow
import traceback
from src.data_processing.data_processing import load_csv, clean_data


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
            #print(f"\n=== Debug inicio ===")
            #print(f"Ruta del archivo: {file_path}")

            # 1. Carga el archivo
            self.df = load_csv(file_path)  # Carga el archivo CSV
            # debug para reconocer la carga del archivo
            # print(f"DataFrame cargado. Filas: {len(self.df)}")
            # print(self.df.head(2))
           
                        # Actualiza la vista previa
            numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            self.preview_x_dropdown.options = [ft.dropdown.Option(col) for col in numeric_cols]
            self.preview_y_dropdown.options = [ft.dropdown.Option(col) for col in numeric_cols]
            
            if numeric_cols:
               self.preview_x_dropdown.value = numeric_cols[0]
               self.preview_y_dropdown.value = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] 
               
            #habilita los dropdowns
            self.preview_x_dropdown.disabled = False
            self.preview_y_dropdown.disabled = False
            self.update_preview_btn.disabled = False
            self.page.update()
            
            self.controls[0].controls[0].controls[-1].disabled = False
             # 3. Asignación solo si todo está bien
            # print(f"Columnas válidas: {self.df.columns.tolist()}")
            
            self.show_snackbar("Archivo cargado correctamente", "green")
            # 4. Actualización UI
            self.update_preview(e)
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
            
    def update_preview(self, e):
        try:
            if self.df is None:
                raise ValueError("No hay datos cargados")
            
            x_col = self.preview_x_dropdown.value
            y_col = self.preview_y_dropdown.value

            if not x_col or not y_col:
                raise ValueError("Selecciona ambas columnas.")
            
            # Debug: verifica valores
            # print(f"Columnas seleccionadas: X={x_col}, Y={y_col}")
            # print(self.df[[x_col, y_col]].head(2))            
            
            self.update_data_table(x_col, y_col)
            self.show_snackbar("Vista previa actualizada", "green")
            
        except Exception as ex:
            print(f"Error en update_preview: {traceback.format_exc()}")
            self.show_snackbar(f"Error al mostrar vista previa: {str(ex)}", "red")
            
    def update_data_table(self, x_col=None , y_col=None):
        try:
            if self.df is None or self.df.empty:
                return
            # Si se especifican columnas, muestra solo esas
            if x_col and y_col:
                display_df = self.df[[x_col, y_col]].head(5)
            else:
                display_df = self.df.head(5)
            # Crear columnas para el DataTable
            columns = [ft.DataColumn(ft.Text(col)) for col in display_df.columns[:5]]  # Mostrar máximo 5 columnas
        
            # Crear filas con los primeros registros
            rows = []
            for _, row in display_df.iterrows():  
                cells = [ft.DataCell(ft.Text(str(row[col]))) for col in display_df.columns[:5]]
                rows.append(ft.DataRow(cells=cells))
        
            self.data_preview.columns = columns
            self.data_preview.rows = rows
            self.page.update()
        
        except Exception as ex:
            print(f"Error actualizando tabla: {str(ex)}")
            self.show_snackbar("Error al mostrar datos", "red")
            raise
            
    def apply_cleaning(self, e):
        
        try:
            if self.df is None or self.df.empty:
                raise ValueError("No hay datos cargados para limpiar")

            # Mostrar indicador de procesamiento
            #self.page.splash = ft.ProgressBar()
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
            
            offline_win = OfflineWindow(self.page, cleaned_df)
            self.page.add(offline_win)
            
            self.page.update()
        
        except Exception as ex:
            self.show_snackbar(f"Error en limpieza: {str(ex)}", "red")
        
        finally:
            # Asegurarse de quitar el indicador de carga
            #self.page.splash = None
            self.page.update()
            
    def show_snackbar(self, message: str, color: str = "green"):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor = color)
        self.page.snack_bar.open = True
        self.page.update()

