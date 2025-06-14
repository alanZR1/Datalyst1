import flet as ft
from interface.Offline_Window import OfflineWindow
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
            "fill_na": "none",
            "remove_duplicates": False,
            "normalize": False
        }
        
        self.back_button = ft.FloatingActionButton(
            icon = "arrow_back",
            bgcolor = ft.Colors.GREY_900,
            shape = ft.CircleBorder(),
            autofocus = True,
            tooltip = "Volver al inicio",
            mini = True,
            on_click = self.go_back
        )
        

        self.continue_btn = ft.ElevatedButton(
            "Continuar a Entrenamiento",
            on_click = self.apply_cleaning,
            disabled = True,
            width = 200
        )


        self.preview_x_dropdown = ft.Dropdown(
            label = "Vista previa - Eje X", 
            options = [], 
            disabled = True,
            width = 200
        )
        
        
        self.preview_y_dropdown = ft.Dropdown(
            label = "Vista previa - Eje Y", 
            options = [], 
            disabled = True,
            width = 200
        )
        

        self.update_preview_btn = ft.ElevatedButton(
            "Actualizar Vista Previa",
            on_click = self.update_preview,
            disabled = True,
            width = 200
        )
        

        self.file_picker = ft.FilePicker(on_result = self.file_selected)
        self.page.overlay.append(self.file_picker)
        
        self.remove_outliers = ft.Switch(label = "Eliminar Outliers", value = False)
        
        self.fill_na = ft.Dropdown(
            label = "Rellenar valores nulos",
            options = [
                ft.dropdown.Option("none", "No rellenar"),
                ft.dropdown.Option("mean", "Media"),
                ft.dropdown.Option("median", "Mediana"),
            ],
            
            value = "none"
        
        )

        self.normalize_data = ft.Switch(label = "Normalizar datos", value = False)
        self.remove_duplicates = ft.Switch(label = "Eliminar duplicados", value = False)
        self.data_preview = ft.DataTable(
            columns = [ft.DataColumn(ft.Text("Cargando datos..."))],
            rows = []
        )
        
        
        self.controls = [
            
            ft.Column([                           
                ft.Row([
                        ft.Column([
                                ft.Row([
                                    
                                    ft.Container(
                                        content = self.back_button,
                                        alignment = ft.alignment.center_right,
                                        expand = True
                                    ),
                                    ft.Text("Preprocesamiento de Datos", size = 18, weight = "bold"),
                                ], 
                                alignment = "spaceBetween"
                                ),
                                
                                ft.Divider(height = 20),
                                ft.ElevatedButton("Cargar CSV", on_click = self.pick_file),
                                self.preview_x_dropdown,
                                self.preview_y_dropdown,
                                
                                ft.Text("Opciones de limpieza:", size = 16),
                                self.remove_outliers,
                                self.fill_na,
                                self.normalize_data,
                                self.remove_duplicates,
                                ft.Divider(height = 20),
                                self.continue_btn,
                                
                                
                            ],
                            width = 300
                        ),
                        
                        
                        ft.Container(
                            content = ft.Column([
                                ft.Text("Vista previa de datos:", size = 16),
                                ft.Container(
                                    content = self.data_preview,
                                    border = ft.border.all(1, ft.Colors.GREY_400),
                                    padding = 30,
                                    border_radius = 5,
                                    width = 500,
                                ),
                            self.update_preview_btn,
                            ]),
                            
                        expand = False,
                        
                        ),
                        
                    ],
                       
                    expand = True
                
                ),
            ]),
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
            

            self.df = load_csv(file_path)  
            print(f"DataFrame cargado. Filas: {len(self.df)}")
            print(self.df.head(2))
            
            
            numeric_cols = self.df.select_dtypes(include = ["number"]).columns.tolist()
            
            self.preview_x_dropdown.options = [ft.dropdown.Option(col) for col in numeric_cols]
            self.preview_y_dropdown.options = [ft.dropdown.Option(col) for col in numeric_cols]

            if numeric_cols:
               self.preview_x_dropdown.value = numeric_cols[0]
               self.preview_y_dropdown.value = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] 

            self.preview_x_dropdown.disabled = False
            self.preview_y_dropdown.disabled = False
            self.update_preview_btn.disabled = False
            self.page.update()

            self.controls[0].controls[0].controls[-1].disabled = False
            print(f"Columnas válidas: {self.df.columns.tolist()}")
            

            self.show_snackbar("Archivo cargado correctamente", "green")
            
            self.update_preview(e)
            self.enable_training_button()

            print("=== Debug fin ===\n")

        except Exception as ex:
            
            error_msg = f"Error al cargar {file_path}: {str(ex)}"
            
            print(f"\n!!! ERROR: {error_msg}")
            print(traceback.format_exc())
            
            self.show_snackbar(error_msg, "red")
            self.df = None  


    def enable_training_button(self):
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

            
            print(f"Columnas seleccionadas: X = {x_col}, Y = {y_col}")
            print(self.df[[x_col, y_col]].head(2))            
                        

            self.update_data_table(x_col, y_col)
            self.show_snackbar("Vista previa actualizada", "green")

        except Exception as ex:
            print(f"Error en update_preview: {traceback.format_exc()}")
            self.show_snackbar(f"Error al mostrar vista previa: {str(ex)}", "red")

    def update_data_table(self, x_col = None , y_col = None):
        try:
            if self.df is None or self.df.empty:
                return
            
            if x_col and y_col:
                display_df = self.df[[x_col, y_col]].head(5)
            else:
                display_df = self.df.head(5)
            
            columns = [ft.DataColumn(ft.Text(col)) for col in display_df.columns[:5]] 

            rows = []
            for _, row in display_df.iterrows():  
                cells = [ft.DataCell(ft.Text(str(row[col]))) for col in display_df.columns[:5]]
                rows.append(ft.DataRow(cells = cells))

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

            self.page.splash = ft.ProgressBar()
            self.page.update()
            print ("Aplicando limpieza de datos...")

            cleaned_df = clean_data(
                df = self.df,
                remove_outliers = self.remove_outliers.value,
                fill_na = self.fill_na.value,
                remove_duplicates = self.remove_duplicates.value,
                normalize = self.normalize_data.value
            )
            
            print(f"Limpieza completada.")

            if cleaned_df.empty:
                raise ValueError("El DataFrame quedó vacío después de la limpieza")

            self.page.clean()
            offline_win = OfflineWindow(self.page, cleaned_df)
            self.page.add(offline_win)
            self.page.update()
        
        except Exception as ex:
            self.show_snackbar(f"Error en limpieza: {str(ex)}", "red")
        
        finally:
            self.page.splash = None
            self.page.update()
            
            
            
    def go_back(self, e):
        self.page.overlay.clear()
        from interface.main_app import MainApp
        self.page.clean()
        self.page.add(MainApp(self.page))
        self.page.update()
            
            
    def show_snackbar(self, message: str, color: str = "green"):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor = color)
        self.page.snack_bar.open = True
        self.page.update()