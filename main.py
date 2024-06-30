import tkinter as tk
from tkinter import Canvas, Scale, HORIZONTAL, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from predictalg import predict

class MapEditor:
    def __init__(self, root, veg_image_path, temp_image_path, temp_slider_path, popdensity_image_path, water_image_path, cityborder_image_path):
        self.root = root
        self.root.title("NYC Temperature Map Predictor")
        
        self.veg_image = cv2.imread(veg_image_path)
        self.veg_image = cv2.cvtColor(self.veg_image, cv2.COLOR_BGR2RGB)
        
        target_width = self.veg_image.shape[1]
        target_height = self.veg_image.shape[0]
        
        self.temp_image = self.load_and_resize_image(temp_image_path, target_width, target_height)
        self.popdensity_image = self.load_and_resize_image(popdensity_image_path, target_width, target_height)
        self.water_image = self.load_and_resize_image(water_image_path, target_width, target_height)
        self.cityborder_image = self.load_and_resize_image(cityborder_image_path, target_width, target_height)
        
        self.current_image = self.veg_image
        self.image_pil = Image.fromarray(self.current_image)
        self.image_tk = ImageTk.PhotoImage(self.image_pil)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.LEFT)

        self.canvas = Canvas(self.canvas_frame, width=self.image_pil.width, height=self.image_pil.height)
        self.canvas.pack()
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
        
        self.controls_frame = tk.Frame(root, width=200, height=self.image_pil.height)
        self.controls_frame.pack_propagate(False)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.layers_label = Label(self.controls_frame, text="Layers:")
        self.layers_label.pack(pady=(20, 10))

        self.vegetation_button = Button(self.controls_frame, text="Vegetation", command=self.show_vegetation)
        self.vegetation_button.pack(pady=(5, 10))

        self.temperature_button = Button(self.controls_frame, text="Temperature", command=self.show_temperature)
        self.temperature_button.pack(pady=(5, 10))

        self.popdensity_button = Button(self.controls_frame, text="Population Density", command=self.show_popdensity)
        self.popdensity_button.pack(pady=(5, 10))

        self.water_button = Button(self.controls_frame, text="Water", command=self.show_water)
        self.water_button.pack(pady=(5, 10))

        self.cityborder_button = Button(self.controls_frame, text="City Border", command=self.show_cityborder)
        self.cityborder_button.pack(pady=(5, 20))

        self.brush_radius = 5
        self.brush_color = (0, 255, 0)

        self.radius_slider = Scale(self.controls_frame, from_=1, to=50, orient=HORIZONTAL, label='Brush Radius')
        self.radius_slider.pack(pady=10)
        self.radius_slider.set(self.brush_radius)

        self.color_scale_image = self.create_color_scale_image()
        self.color_scale = np.array(self.color_scale_image)
        self.color_pil = Image.fromarray(self.color_scale)
        self.color_tk = ImageTk.PhotoImage(self.color_pil)
        
        self.color_slider = Scale(self.controls_frame, from_=0, to=self.color_scale.shape[1]-1, orient=HORIZONTAL, label='Color Intensity')
        self.color_slider.pack(pady=10)
        self.color_slider.set(self.color_scale.shape[1]//2)

        self.color_canvas = Canvas(self.controls_frame, width=self.color_pil.width, height=self.color_pil.height)
        self.color_canvas.pack(pady=10)
        self.color_canvas.create_image(0, 0, anchor="nw", image=self.color_tk)
        
        self.veg_slider_label_frame = tk.Frame(self.controls_frame)
        self.veg_slider_label_frame.pack()
        
        self.sparse_label = Label(self.veg_slider_label_frame, text="sparse")
        self.sparse_label.pack(side=tk.LEFT, padx=(0, 75))
        
        self.dense_label = Label(self.veg_slider_label_frame, text="dense")
        self.dense_label.pack(side=tk.RIGHT, padx=(35, 0))
        
        self.popdensity_slider_label_frame = tk.Frame(self.controls_frame)
        
        self.popdensity_min_label = Label(self.popdensity_slider_label_frame, text="0 ppl/mi²")
        self.popdensity_min_label.pack(side=tk.LEFT, padx=(0, 75))
        
        self.popdensity_max_label = Label(self.popdensity_slider_label_frame, text="250 ppl/mi²")
        self.popdensity_max_label.pack(side=tk.RIGHT, padx=(0, 0))

        self.water_key_frame = tk.Frame(self.controls_frame)
        self.water_key_frame.pack(pady=(10, 0))

        self.water_color_box = Canvas(self.water_key_frame, width=20, height=20, bg="white")
        self.water_color_box.pack(side=tk.LEFT, padx=(0, 10))

        self.water_label = Label(self.water_key_frame, text="Water")
        self.water_label.pack(side=tk.LEFT)
        
        self.water_key_frame.pack_forget()
        
        self.border_key_frame = tk.Frame(self.controls_frame)
        self.border_key_frame.pack(pady=(10, 0))

        self.border_color_box = Canvas(self.border_key_frame, width=20, height=20, bg="white")
        self.border_color_box.pack(side=tk.LEFT, padx=(0, 10))

        self.border_label = Label(self.border_key_frame, text="City Land")
        self.border_label.pack(side=tk.LEFT)
        
        self.border_key_frame.pack_forget()

        # Load and resize the temperature slider image
        self.temp_slider_image = Image.open(temp_slider_path)
        self.temp_slider_image = self.temp_slider_image.resize((180, 20), Image.Resampling.LANCZOS)
        self.temp_slider_tk = ImageTk.PhotoImage(self.temp_slider_image)

        self.temp_slider_canvas = Canvas(self.controls_frame, width=180, height=20)
        self.temp_slider_canvas.create_image(0, 0, anchor="nw", image=self.temp_slider_tk)

        self.temp_slider_label_frame = tk.Frame(self.controls_frame)

        self.warm_label = Label(self.temp_slider_label_frame, text="warm")
        self.warm_label.pack(side=tk.LEFT, padx=(0, 75))
        
        self.hot_label = Label(self.temp_slider_label_frame, text="hot")
        self.hot_label.pack(side=tk.RIGHT, padx=(60, 0))

        self.apply_button = Button(self.controls_frame, text="Apply", command=self.apply_changes)
        self.apply_button.pack(pady=(20, 10), side=tk.BOTTOM, anchor="center")

        self.bind_paint_events()
    
    def load_and_resize_image(self, image_path, width, height):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image

    def create_color_scale_image(self):
        width = 300
        height = 20
        color_scale_image = np.zeros((height, width, 3), dtype=np.uint8)
        start_color = np.array([255, 255, 255])  # White
        end_color = np.array([11, 38, 0])  # #0B2600
        for i in range(width):
            ratio = i / width
            color = start_color * (1 - ratio) + end_color * ratio
            color_scale_image[:, i] = color
        return Image.fromarray(color_scale_image.astype(np.uint8))
    
    def create_popdensity_color_scale_image(self):
        width = 300
        height = 20
        color_scale_image = np.zeros((height, width, 3), dtype=np.uint8)
        start_color = np.array([0, 0, 0])  # Black
        end_color = np.array([255, 255, 255])  # White
        for i in range(width):
            ratio = i / width
            color = start_color * (1 - ratio) + end_color * ratio
            color_scale_image[:, i] = color
        return Image.fromarray(color_scale_image.astype(np.uint8))

    def paint(self, event):
        x, y = event.x, event.y
        print(f"Mouse event at ({x}, {y})")
        if 0 <= x < self.image_pil.width and 0 <= y < self.image_pil.height:
            self.brush_radius = self.radius_slider.get()
            color_index = self.color_slider.get()
            self.brush_color = tuple(map(int, self.color_scale[0, color_index]))
            print(f"Painting at ({x}, {y}) with radius {self.brush_radius} and color {self.brush_color}")
            cv2.circle(self.current_image, (x, y), self.brush_radius, self.brush_color, -1)
            self.update_canvas()
        else:
            print("Mouse event out of bounds")

    def paint_water(self, event):
        x, y = event.x, event.y
        print(f"Mouse event at ({x}, {y})")
        if 0 <= x < self.image_pil.width and 0 <= y < self.image_pil.height:
            self.brush_radius = self.radius_slider.get()
            self.brush_color = (255, 255, 255)  # White color for water
            print(f"Painting at ({x}, {y}) with radius {self.brush_radius} and color {self.brush_color}")
            cv2.circle(self.current_image, (x, y), self.brush_radius, self.brush_color, -1)
            self.update_canvas()
        else:
            print("Mouse event out of bounds")

    def update_canvas(self):
        self.image_pil = Image.fromarray(self.current_image)
        self.image_tk = ImageTk.PhotoImage(self.image_pil)
        self.canvas.itemconfig(self.image_on_canvas, image=self.image_tk)
        self.canvas.image = self.image_tk  # Keep a reference to avoid garbage collection
        print("Canvas updated")
    
    def apply_changes(self):
        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
        gray_3_channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        
        pdensity = np.array(self.popdensity_image)[:,:,0] / 255 * 2.5 # measured in hundreds of thousands per sq mile
        vegetation = np.array(gray_3_channel)[:,:,0] / 255 * 0.8
        water = np.array(self.water_image)[:,:,0] / 255
        applicable = np.array(self.cityborder_image)[:,:,0] / 255
        print(pdensity.shape)
        print(vegetation.shape)
        print(water.shape)
        print(applicable.shape)
        
        image = predict(pdensity, vegetation, water, applicable)    
        self.temp_image = np.array(image)

    def resize_image(self, image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        return cv2.resize(image, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)

    def show_vegetation(self):
        print("Vegetation layer selected")
        self.current_image = self.veg_image.copy()

        self.radius_slider.pack(pady=10)
        self.color_slider.pack(pady=10)
        
        # Update color scale to green-to-white
        self.color_scale_image = self.create_color_scale_image()
        self.color_scale = np.array(self.color_scale_image)
        self.color_pil = Image.fromarray(self.color_scale)
        self.color_tk = ImageTk.PhotoImage(self.color_pil)
        
        self.color_canvas.create_image(0, 0, anchor="nw", image=self.color_tk)
        self.color_canvas.pack(pady=10)
        
        self.veg_slider_label_frame.pack(pady=(10, 0))
        self.temp_slider_canvas.pack_forget()
        self.temp_slider_label_frame.pack_forget()
        self.popdensity_slider_label_frame.pack_forget()
        self.water_key_frame.pack_forget()
        self.border_key_frame.pack_forget()
        self.apply_button.pack(pady=(20, 10), side=tk.BOTTOM, anchor="center")
        self.update_canvas()
        self.bind_paint_events()


    def show_temperature(self):
        print("Temperature layer selected")
        self.current_image = self.temp_image.copy()
        self.current_image = self.resize_image(self.current_image)

        self.radius_slider.pack_forget()
        self.color_canvas.pack_forget()
        self.color_slider.pack_forget()
        self.veg_slider_label_frame.pack_forget()
        self.popdensity_slider_label_frame.pack_forget()
        self.water_key_frame.pack_forget()
        self.apply_button.pack_forget()
        self.border_key_frame.pack_forget()
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<Button-1>")
        
        self.temp_slider_canvas.pack(pady=(10, 0))
        self.temp_slider_label_frame.pack(pady=(0, 10))
        self.temp_slider_canvas.create_image(0, 0, anchor="nw", image=self.temp_slider_tk)
        self.update_canvas()

    def show_popdensity(self):
        print("Population Density layer selected")
        self.current_image = self.popdensity_image.copy()
        self.current_image = self.resize_image(self.current_image)
        self.radius_slider.pack(pady=10)
        self.color_slider.pack(pady=10)
        
        self.color_scale_image = self.create_popdensity_color_scale_image()
        self.color_scale = np.array(self.color_scale_image)
        self.color_pil = Image.fromarray(self.color_scale)
        self.color_tk = ImageTk.PhotoImage(self.color_pil)
        
        self.color_canvas.create_image(0, 0, anchor="nw", image=self.color_tk)
        self.color_canvas.pack(pady=10)

        self.popdensity_slider_label_frame.pack(pady=(10, 0))
        self.temp_slider_canvas.pack_forget()
        self.temp_slider_label_frame.pack_forget()
        self.veg_slider_label_frame.pack_forget()
        self.water_key_frame.pack_forget()
        self.border_key_frame.pack_forget()
        self.apply_button.pack(pady=(20, 10), side=tk.BOTTOM, anchor="center")
        self.update_canvas()
        self.bind_paint_events()


    def show_water(self):
        print("Water layer selected")
        self.current_image = self.water_image.copy()
        self.current_image = self.resize_image(self.current_image)
        self.radius_slider.pack(pady=10)
        self.color_canvas.pack_forget()
        self.color_slider.pack_forget()
        self.veg_slider_label_frame.pack_forget()
        self.temp_slider_canvas.pack_forget()
        self.temp_slider_label_frame.pack_forget()
        self.popdensity_slider_label_frame.pack_forget()
        self.border_key_frame.pack_forget()
        
        self.water_key_frame.pack(pady=(10, 0))
        
        self.apply_button.pack(pady=(20, 10), side=tk.BOTTOM, anchor="center")
        self.bind_paint_events_water()
        self.update_canvas()

    def show_cityborder(self):
        print("City Border layer selected")
        self.current_image = self.cityborder_image.copy()
        self.current_image = self.resize_image(self.current_image)
        self.radius_slider.pack(pady=10)
        self.color_canvas.pack_forget()
        self.color_slider.pack_forget()
        self.veg_slider_label_frame.pack_forget()
        self.temp_slider_canvas.pack_forget()
        self.temp_slider_label_frame.pack_forget()
        self.popdensity_slider_label_frame.pack_forget()
        self.water_key_frame.pack_forget()
        
        self.border_key_frame.pack(pady=(10, 0))
        
        self.apply_button.pack(pady=(20, 10), side=tk.BOTTOM, anchor="center")      
        self.bind_paint_events_water()
        self.update_canvas()

    def bind_paint_events(self):
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        print("Paint events bound")

    def bind_paint_events_water(self):
        self.canvas.bind("<B1-Motion>", self.paint_water)
        self.canvas.bind("<Button-1>", self.paint_water)
        print("Water paint events bound")

if __name__ == "__main__":
    root = tk.Tk()
    app = MapEditor(root, 'newyork_veg_2002226_lrg.jpg', 'newyork_tem_2002226_lrg.jpg', 'slider.png', 'popdensity.png', 'water.png', 'applicable_nojfk.png')
    root.mainloop()
