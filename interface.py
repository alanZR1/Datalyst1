import flet as ft
from data_processing import load_csv
from clustering import train_kmeans, calculate_silhouette
from model_save_carge import save_model, load_model


class MainApp(ft.Column):
    def __init__(self, page):
        super().__init__(scroll=ft.ScrollMode.AUTO)
        self.page = page
        self.data = None
        self.kmeans = None
        self.file_picker = ft.FilePicker()
        self.page.overlay.append(self.file_picker)
        self.file_picker.on_result = self.file_selected

        self.k_input = ft.TextField(label="Número de clusters", value="0")
        self.n_init_input = ft.TextField(label="Número de iteraciones", value="0")
        self.silhouette_text = ft.Text("Índice de silueta: ")
        self.image = ft.Image(width=500, height=400)
        self.silhouette_button = ft.ElevatedButton("Calcular Silueta", on_click=self.calculate_silhouette, visible=False)
        self.classification_result = ft.Text("Resultado de Clasificación: ")
        self.history = ft.ListView(expand=True)

        self.x_axis_dropdown = ft.Dropdown(label="Característica X", options=[])
        self.y_axis_dropdown = ft.Dropdown(label="Característica Y", options=[])

        self.training_method = ft.Dropdown(
            label="Método de entrenamiento",
            options=[ft.dropdown.Option("K-Means"), ft.dropdown.Option("Otro método")]
        )

        self.controls = [
            ft.Row([
                ft.Column([
                    ft.Text("Selecciona un modo:"),
                    ft.ElevatedButton("Offline", on_click=self.pick_file),
                    ft.ElevatedButton("Online", on_click=self.open_online_window),
                    ft.Text("Entrenamiento"),
                    self.training_method,
                    self.k_input,
                    self.n_init_input,
                    self.x_axis_dropdown,
                    self.y_axis_dropdown,
                    ft.ElevatedButton("Entrenar", on_click=self.train_kmeans, visible=False),
                    ft.ElevatedButton("Guardar Modelo", on_click=self.save_model, visible=False),
                    ft.ElevatedButton("Cargar Modelo", on_click=self.load_model),
                    self.silhouette_button,
                    self.silhouette_text,
                    self.classification_result,
                    self.history
                ], expand=1),
                ft.Container(content=self.image, expand=2)
            ])
        ]

    def pick_file(self, e):
        self.file_picker.pick_files(allow_multiple=False)

    def file_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            self.data = e.files[0].path
            df = load_csv(self.data) # ya contiene los datos limpios
            
            numeric_columns = df.select_dtypes(include=["number"]).columns
            self.x_axis_dropdown.options = [ft.dropdown.Option(str(i)) for i in range(len(numeric_columns))]
            self.y_axis_dropdown.options = [ft.dropdown.Option(str(i)) for i in range(len(numeric_columns))]

            self.controls[0].controls[0].controls[9].visible = True
            self.page.update()

    def train_kmeans(self, e):
        try:
            k = int(self.k_input.value)
            n_init = int(self.n_init_input.value)
            img_base64, self.kmeans = train_kmeans(self.data, k, n_init, self.x_axis_dropdown.value, self.y_axis_dropdown.value)
            self.image.src_base64 = img_base64
            self.silhouette_button.visible = True
            self.controls[0].controls[0].controls[10].visible = True
            self.page.update()
        except Exception as ex:
            print(f"Error en entrenamiento: {ex}")

    def calculate_silhouette(self, e):
        score = calculate_silhouette(self.data, self.kmeans)
        self.silhouette_text.value = f"Índice de silueta: {score:.4f}"
        self.page.update()

    def save_model(self, e):
        save_model(self.kmeans, "kmeans_model.pkl")
        self.silhouette_text.value = "✅ Modelo guardado exitosamente"
        self.page.update()

    def load_model(self, e):
        self.kmeans = load_model("kmeans_model.pkl")
        self.silhouette_text.value = "✅ Modelo cargado exitosamente"
        self.page.update()

    def open_online_window(self, e):
        print("Modo Online seleccionado")
        self.page.snack_bar = ft.SnackBar(ft.Text("Modo Online aún no implementado"), bgcolor="red")
        self.page.snack_bar.open = True
        self.page.update()
