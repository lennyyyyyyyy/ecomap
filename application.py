import tkinter as tk
from tkinter import Canvas, Scale, HORIZONTAL, Label, Button, Spinbox
from PIL import Image, ImageTk
import cv2
import numpy as np
from predictalg import predict
import math

class ChooseScreen(tk.Frame):
    def __init__(self):
        super().__init__(root)
        Label(self, text="Choose a preset:").pack()
        Button(self, text="New York City", command=self.choose_nyc).pack()
        Button(self, text="Custom", command=self.choose_custom).pack()
        self.widthInput = Spinbox(self, from_=0, to=2000, increment=10, text="Width")
        self.widthInput.pack()
        self.heightInput = Spinbox(self, from_=0, to=2000, increment=10, text="Height")
        self.heightInput.pack()
        self.pack()
    def choose_nyc(self):
        self.pack_forget()
        app = MapEditor("NYC")
    def choose_custom(self):
        self.pack_forget()
        app = MapEditor("Custom", width=int(self.widthInput.get()), height=int(self.heightInput.get()))

class MapEditor(tk.Frame):
    instance = None
    def __init__(self, preset, width=0, height=0):
        MapEditor.instance = self
        super().__init__(root)
        if preset == "NYC": # account for presets
            self.pdensity = np.array(Image.open("popdensity.png"))[:,:,:3]
            self.vegetation = np.array(Image.open("vegetation.png"))[:,:,:3]
            self.water = np.array(Image.open("water.png"))[:,:,:3]
            self.applicable = np.array(Image.open("applicable_nojfk.png"))[:,:,:3]
            self.width = 2345
            self.height = 2334
        else:
            self.pdensity = np.zeros((width, height, 3), dtype=np.uint8)
            self.vegetation = np.zeros((width, height, 3), dtype=np.uint8)
            self.water = np.zeros((width, height, 3), dtype=np.uint8)
            self.applicable = np.zeros((width, height, 3), dtype=np.uint8)
            self.width = width
            self.height = height
        self.temps = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        self.canvas_frame = tk.Frame(self) #is this really necessary
        self.canvas_frame.pack(side=tk.LEFT)

        if self.width > self.height:
            self.scaled_width = 800
            self.scaled_height = int(self.height * 800 / self.width)
        else:
            self.scaled_height = 800
            self.scaled_width = int(self.width * 800 / self.height)

        self.canvas = MainCanvas(self.canvas_frame, self.scaled_width, self.scaled_height)


        self.controls_frame = tk.Frame(self, width=300, height=800)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

        Button(self.controls_frame, text="Evaluate and Predict", command=self.predict).pack(pady=(20, 10))

        Label(self.controls_frame, text="Input Layers:").pack(pady=(20, 10))

        Button(self.controls_frame, text="Vegetation", command=self.show_vegetation).pack(pady=(5, 10))

        Button(self.controls_frame, text="Population Density", command=self.show_popdensity).pack(pady=(5, 10))

        Button(self.controls_frame, text="Water", command=self.show_water).pack(pady=(5, 10))

        Button(self.controls_frame, text="City Border", command=self.show_cityborder).pack(pady=(5, 20))

        Label(self.controls_frame, text="Output Layers:").pack(pady=(20, 10))

        Button(self.controls_frame, text="Temperature", command=self.show_temperature).pack(pady=(5, 10))

        self.vegetation_controls = BrushControls(self.controls_frame, 0, 100, 1, "Percent Vegetative Cover")
        self.pdensity_controls = BrushControls(self.controls_frame, 0, 250, 1, "Thousands per sq mile")
        self.water_controls = BrushControls(self.controls_frame, 0, 100, 100, "Percent Water Cover")
        self.applicable_controls = BrushControls(self.controls_frame, 0, 1, 1, "City Bounds")

        self.vegetation_controls.pack(pady=(20, 10))
        
        self.currLayer = self.vegetation
        self.canvas.update()
        self.currControls = self.vegetation_controls

        self.pack()
        
    def show_vegetation(self):
        self.vegetation_controls.pack(pady=(20, 10))
        self.currControls.pack_forget()
        self.currControls = self.vegetation_controls
        self.currLayer = self.vegetation
        self.canvas.update()
    def show_temperature(self):
        self.currControls.pack_forget()
        self.currLayer = self.temps
        self.canvas.update()
    def show_popdensity(self):
        self.pdensity_controls.pack(pady=(20, 10))
        self.currControls.pack_forget()
        self.currControls = self.pdensity_controls
        self.currLayer = self.pdensity
        self.canvas.update()
    def show_water(self):
        self.water_controls.pack(pady=(20, 10))
        self.currControls.pack_forget()
        self.currControls = self.water_controls
        self.currLayer = self.water
        self.canvas.update()
    def show_cityborder(self):
        self.applicable_controls.pack(pady=(20, 10))
        self.currControls.pack_forget()
        self.currControls = self.applicable_controls
        self.currLayer = self.applicable
        self.canvas.update()
    def predict(self):
        prediction = predict(self.pdensity[:, :, 0] * 2.5 / 255, self.vegetation[:, :, 0] / 255, self.water[:, :, 0] / 255, self.applicable[:, :, 0] / 255, 23)
        self.temps = np.repeat(12*(prediction+10), 3, axis=-1)
        self.show_temperature()
class MainCanvas(Canvas):
    def __init__(self, parent, width, height):
        super().__init__(parent, width=width, height=height)
        self.width = width
        self.height = height
        self.pack()

        self.bind("<B1-Motion>", self.paint)
        self.bind("<Button-1>", self.paint)
    def update(self):
        self.delete("all")
        self.image = ImageTk.PhotoImage(Image.fromarray(cv2.resize(MapEditor.instance.currLayer, (self.width, self.height))))
        self.create_image(0, 0, image=self.image, anchor="nw")
    def paint(self, event):
        x, y = event.x, event.y
        if not(x < 0 or x >= self.width or y < 0 or y >= self.height):
            convx = int(x * MapEditor.instance.width / self.width)
            convy = int(y * MapEditor.instance.height / self.height)
            brushSize = MapEditor.instance.currControls.getBrushSize()
            opacity = MapEditor.instance.currControls.getOpacity()
            radius = brushSize // 2
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    if convx+i >= 0 and convx+i < MapEditor.instance.width and convy+j >= 0 and convy+j < MapEditor.instance.height and math.hypot(i, j) <= radius:
                        MapEditor.instance.currLayer[convy+j, convx+i] = np.array([opacity, opacity, opacity])*255
            self.update()


class BrushControls(tk.Frame):
    def __init__(self, parent, from_, to, resolution, rangeLabel):
        super().__init__(parent)
        self.brush_size = Scale(self, from_=1, to=50, orient=HORIZONTAL, label="Brush Size", length=250)
        self.brush_size.pack(pady=(20, 10))
        self.opacity = Scale(self, from_=from_, to=to, resolution=resolution, orient=HORIZONTAL, label=rangeLabel, length=250)
        self.opacity.pack(pady=(10, 20))
    def getBrushSize(self):
        return self.brush_size.get()
    def getOpacity(self):
        return self.opacity.get()/self.opacity.cget("to")







if __name__ == "__main__":
    root = tk.Tk()
    root.title = "NYC Temperature Map Predictor"
    c = ChooseScreen()
    root.mainloop()