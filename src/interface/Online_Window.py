import flet as ft
import pickle
import numpy as np
import random
import threading
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io
import base64


class OnlineWindow(ft.Column):
    def __init__(self, page):
        try:
            super().__init__(expand=True)
            self.page = page
            self.model = None
            self.data = None
            self.x_col = None
            self.y_col = None
            self.running = False

            self.load_model_button = ft.ElevatedButton(
                text="Cargar Modelo",
                on_click=self.pick_model_file
            )

            self.start_button = ft.ElevatedButton(
                text = "Iniciar Lectura",
                on_click = self.start_streaming,
                disabled = True
            )

            self.stop_button = ft.ElevatedButton(
                text = "Detener Lectura",
                on_click = self.stop_streaming,
                disabled = True
            )
            
            self.back_button = ft.FloatingActionButton(
                icon = "Arrow_Back",
                bgcolor = ft.Colors.GREY_900,
                shape = ft.CircleBorder(),
                autofocus = True,
                tooltip = "Volver",
                mini = True,
                on_click = self.go_back
            )
            

            self.cluster_text = ft.Text("Cluster: -", size = 14)
            self.status_text = ft.Text("Estado: -", size = 16, weight = "bold")

            
            self.cluster_image = ft.Image(
                width = 600,
                height = 400,
                border_radius = 10,
                fit = ft.ImageFit.CONTAIN)  
            

            self.file_picker = ft.FilePicker(on_result = self.on_model_selected)
            self.page.overlay.append(self.file_picker)



            self.controls = [
                ft.Stack([
                    ft.Row([
                        ft.Container(
                           content = ft.Column([
                                ft.Text("MONITOREO ONLINE", size = 18, weight = "bold"),
                                self.load_model_button,
                                self.start_button,
                                self.stop_button,
                                self.cluster_text,
                                self.status_text,
                            ],
                            spacing = 10 
                           ),
                           
                            width = 300,
                            padding = 15,                  
                        ),

                    ft.Container(
                        content = self.cluster_image,
                        padding = 15,
                        expand = True
                    )
                ],
                
                expand = True),
                                     
                    ft.Container(
                        content = self.back_button,
                        alignment = ft.alignment.top_right,
                        padding = 15,
                        margin = ft.margin.only(right = 10, bottom = 10),
                    )
                    
                ],
                expand = True)
                
            ]

            
        except Exception as ex:
            print(f"Error al inicializar OnlineWindow: {str(ex)}")
            


    def pick_model_file(self, e):
        self.file_picker.pick_files(allow_multiple = False)



    def on_model_selected(self, e: ft.FilePickerResultEvent):
        try:
            if e.files:
                path = e.files[0].path
                self.load_model_and_data(path)
                self.start_button.disabled = False
                self.show_snackbar("✅ Modelo cargado correctamente")
        except Exception as ex:
            self.show_snackbar(f"Error al cargar modelo: {str(ex)}", "red")
        self.page.update()



    def load_model_and_data(self, path):
        try:
            
            with open(path, 'rb') as f:
                saved = pickle.load(f)
            
            self.model = saved['model']
            self.data = saved['training_data']
            self.x_col = saved['features']['x_col']
            self.y_col = saved['features']['y_col']
            
            
            required_keys = ['model', 'training_data', 'features']
            if not all(key in saved for key in required_keys):
                raise ValueError("El archivo no tiene el formato esperado")
            
                   
            print(f"Columnas en datos: {self.data.columns}")
            print(f"X_col: {self.x_col}, Y_col: {self.y_col}")
            print(f"Centroides: {self.model.cluster_centers_}")
            
            self.plot_clusters()
            self.cluster_image.update()
            self.start_button.disabled = False
            self.page.update()
            
        except Exception as ex:
            self.show_snackbar(f"Error al cargar modelo: {str(ex)}", "red")
            print(f"Error al cargar modelo: {str(ex)}")
            
            

    def plot_clusters(self):

        """Actualiza el gráfico con los datos cargados"""
        try:
            
            df = self.data
            x_col = self.x_col
            y_col = self.y_col

            
            plt.figure(figsize = (10, 6))

            
            plt.scatter(
                df[x_col], df[y_col],
                c = self.model.predict(df[[x_col, y_col]]),
                cmap = "viridis",
                alpha = 0.6,
                label = "Datos históricos"
            )

            
            centers = self.model.cluster_centers_
            plt.scatter(
                centers[:, 0], centers[:, 1],
                c = "red", marker = "X", s = 200,
                label = "Centroides"
            )

            
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title("Clusters y Centroides\n(Modo Online)")
            plt.legend()
            plt.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format = "png", dpi = 120, bbox_inches = "tight")
            plt.close()

            
            self.cluster_image.src_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            self.page.update()
            

        except Exception as ex:
            self.show_snackbar(f"Error al graficar: {str(ex)}", "red")        

    


    def start_streaming(self, e):
        self.running = True
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.page.update()
        threading.Thread(target = self.read_loop, daemon = True).start()
        
        

    def stop_streaming(self, e):
        self.running = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.page.update()



    def read_loop(self):
        while self.running:
            try:
                x = random.uniform(self.data[self.x_col].min(), self.data[self.x_col].max())
                y = random.uniform(self.data[self.y_col].min(), self.data[self.y_col].max())
                data_point = np.array([[x, y]])

                cluster = self.model.predict(data_point)[0]
                estado = self.interpretar_estado(cluster)

                self.cluster_text.value = f"Cluster: {cluster}"
                self.status_text.value = f"Estado: {estado}"

                self.page.update()
                time.sleep(1)

            except Exception as ex:
                self.show_snackbar(f"Error durante lectura: {str(ex)}", "red")
                break



    def interpretar_estado(self, cluster: int) -> str:
        if cluster in [0, 1]:
            return "Calidad buena"
        elif cluster == 2:
            return "Datos atípicos"
        elif cluster == 3:
            return "Error crítico"
        return "Desconocido"

    
    
    def go_back(self, e):
        self.page.overlay.clear()
        from src.interface.main_app import MainApp
        self.page.clean()
        self.page.add(MainApp(self.page))
        self.page.update()

    

    def show_snackbar(self, msg, color = "green"):
        self.page.snack_bar = ft.SnackBar(ft.Text(msg), bgcolor = color)
        self.page.snack_bar.open = True
        self.page.update()
