import flet as ft
from data_processing import load_csv
from clustering import train_kmeans, calculate_silhouette
from model_save_carge import save_model, load_model


class MainApp(ft.Column):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.data = None
        self.kmeans = None
        self.df = None

        # Contenedor principal para centrar los botones
        self.controls = [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Selecciona un modo:", size=20, weight="bold"),
                        ft.ElevatedButton("Online", on_click=self.open_online_window),
                        ft.ElevatedButton("Offline", on_click=self.open_offline_window),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,  # Centra los elementos verticalmente
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Centra los elementos horizontalmente
                ),
                alignment=ft.alignment.center,  # Centra el contenido del contenedor
                expand=True,  # Ocupa todo el espacio disponible
            )
        ]


    def open_online_window(self, e):
        self.page.snack_bar = ft.SnackBar(ft.Text("Modo Online aún no implementado"), bgcolor="red")
        self.page.snack_bar.open = True
        self.page.update()

    def open_offline_window(self, e):
        # Limpia la ventana principal y abre la ventana secundaria
        self.page.clean()
        OfflineWindow(self.page)
        
        
        
class OfflineWindow(ft.Column):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.df = None
        self.kmeans = None

        # Componentes de la interfaz
        self.file_picker = ft.FilePicker(on_result=self.file_selected)
        self.page.overlay.append(self.file_picker)

        self.k_input = ft.TextField(label="Número de clusters", value=" ")
        self.n_init_input = ft.TextField(label="Número de iteraciones", value=" ")
        self.silhouette_text = ft.Text("Índice de silueta: ")
        self.image = ft.Image(width=500, height=400)
        self.silhouette_button = ft.ElevatedButton("Calcular Silueta", on_click=self.calculate_silhouette, visible=False)
        self.classification_result = ft.Text("Resultado de Clasificación: ")

        self.x_axis_dropdown = ft.Dropdown(label="Característica X", options=[])
        self.y_axis_dropdown = ft.Dropdown(label="Característica Y", options=[])

        # Botones
        self.load_data_button = ft.ElevatedButton("Cargar Datos", on_click=self.pick_file)
        self.train_button = ft.ElevatedButton("Entrenar", on_click=self.train_kmeans, visible=False)
        self.save_model_button = ft.ElevatedButton("Guardar Modelo", on_click=self.save_model, visible=False)

        # Diseño de la ventana secundaria
        self.controls = [
            ft.Row(
                [
                    ft.Column(
                        [
                            self.load_data_button,
                            ft.Text("Parámetros de Entrenamiento:", size=16, weight="bold"),
                            self.k_input,
                            self.n_init_input,
                            self.x_axis_dropdown,
                            self.y_axis_dropdown,
                            self.train_button,
                            self.save_model_button,
                            self.silhouette_button,
                            self.silhouette_text,
                            self.classification_result,
                        ],
                        expand=1,
                    ),
                    ft.Container(content=self.image, expand=2),
                ],
                expand=True,
            )
        ]

        # Actualiza la página
        self.page.add(self)
        self.page.update()

    def pick_file(self, e):
        self.file_picker.pick_files(allow_multiple=False)

    def file_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            try:
                file_path = e.files[0].path
                self.df = load_csv(file_path)

                # Actualiza los dropdowns con las columnas numéricas
                numeric_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
                self.x_axis_dropdown.options = [ft.dropdown.Option(col) for col in numeric_columns]
                self.y_axis_dropdown.options = [ft.dropdown.Option(col) for col in numeric_columns]

                # Habilita el botón de entrenar
                self.train_button.visible = True
                self.page.update()
            except Exception as ex:
                self.show_snackbar(f"Error: {str(ex)}", "red")

    def train_kmeans(self, e):
        try:
            k = int(self.k_input.value)
            n_init = int(self.n_init_input.value)
            x_col = self.x_axis_dropdown.value
            y_col = self.y_axis_dropdown.value

            if x_col not in self.df.columns or y_col not in self.df.columns:
                raise ValueError("Las columnas seleccionadas no existen en el DataFrame.")

            img_base64, self.kmeans = train_kmeans(self.df, k, n_init, x_col, y_col)
            self.image.src_base64 = img_base64
            self.silhouette_button.visible = True
            self.save_model_button.visible = True
            self.page.update()
        except Exception as ex:
            self.show_snackbar(f"Error en entrenamiento: {str(ex)}", "red")

    def calculate_silhouette(self, e):
        try:
            score = calculate_silhouette(self.df, self.kmeans)
            self.silhouette_text.value = f"Índice de silueta: {score:.4f}"
            self.page.update()
        except Exception as ex:
            self.show_snackbar(f"Error al calcular silueta: {str(ex)}", "red")

    def save_model(self, e):
        save_file_dialog = ft.FilePicker(on_result=self.on_save_model)
        self.page.overlay.append(save_file_dialog)
        self.page.update()
        save_file_dialog.save_file()

    def on_save_model(self, e: ft.FilePickerResultEvent):
        if e.path:
            save_model(self.kmeans, e.path)
            self.show_snackbar(f"✅ Modelo guardado en: {e.path}")
            self.page.update()

    def show_snackbar(self, message: str, color: str = "green"):
        self.page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        self.page.snack_bar.open = True
        self.page.update()