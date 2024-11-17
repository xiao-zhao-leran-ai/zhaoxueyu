import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.top_frame = None
        self.center_frame = None
        self.bottom_frame = None
        self.cap = None
        self.play_task = None
        self.image_id = None
        self.current_scale = 1.0
        self.image_offset = [0, 0]
        self.drag_start = None
        self.init_window()

    def init_window(self):
        self.master.title("智慧交通")
        self.master.geometry("800x600")
        self.master.config(bg="#F0F8FF")

        # 创建顶部框架，用于放置按钮
        self.top_frame = tk.Frame(self.master, bg="#F0F8FF")
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # 按钮框架
        self.buttons_frame = tk.Frame(self.top_frame, bg="#F0F8FF")
        self.buttons_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 添加按钮
        self.select_button = tk.Button(
            self.buttons_frame,
            text="选择视频",
            command=self.open_file
        )
        self.select_button.grid(row=0, column=0, padx=(10, 0), pady=5)

        self.detect_image_button = tk.Button(
            self.buttons_frame,
            text="车辆分类",
            command=self.object_detection
        )
        self.detect_image_button.grid(row=0, column=1, padx=(10, 0), pady=5)

        self.detect_video_button = tk.Button(
            self.buttons_frame,
            text="轨迹追踪",
            command=self.Trajectory_tracking
        )
        self.detect_video_button.grid(row=0, column=2, padx=(10, 0), pady=5)

        self.vehicle_count_button = tk.Button(
            self.buttons_frame,
            text="车流量统计",
            command=self.Vehicle_Counting
        )
        self.vehicle_count_button.grid(row=0, column=3, padx=(10, 0), pady=5)

        self.license_plate_button = tk.Button(
            self.buttons_frame,
            text="车牌检测",
            command=self.License_plate_detection
        )
        self.license_plate_button.grid(row=0, column=4, padx=(10, 0), pady=5)

        self.speed_button = tk.Button(
            self.buttons_frame,
            text="速度检测",
            command=self.Speed_detection
        )
        self.speed_button.grid(row=0, column=5, padx=(10, 0), pady=5)

        self.overspeed_button = tk.Button(
            self.buttons_frame,
            text="超速检测",
            command=self.Overspeed_detection
        )
        self.overspeed_button.grid(row=0, column=6, padx=(10, 0), pady=5)

        self.lane_line_button = tk.Button(
            self.buttons_frame,
            text="车道线检测",
            command=self.Lane_line_detection
        )
        self.lane_line_button.grid(row=0, column=7, padx=(10, 0), pady=5)

        self.construction_area_button = tk.Button(
            self.buttons_frame,
            text="驶入施工区域",
            command=self.Construction_area
        )
        self.construction_area_button.grid(row=0, column=8, padx=(10, 0), pady=5)

        # 创建中心框架，用于放置图像显示区域
        self.center_frame = tk.Frame(self.master, bg="white")
        self.center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 图像显示区域
        self.image_display_area = tk.Canvas(self.center_frame, bg="white")
        self.image_display_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.image_display_area.bind("<Configure>", self.on_resize)

        # 创建底部框架，用于放置表格
        self.bottom_frame = tk.Frame(self.master, bg="#F0F8FF")
        self.bottom_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # 表格头部标签
        self.table_header = tk.Label(
            self.bottom_frame,
            text="类别计数",
            font=("Arial", 16),
            anchor=tk.CENTER,
            bg="#F0F8FF"
        )
        self.table_header.pack(pady=10)

        # 表格
        self.table_body = tk.Listbox(self.bottom_frame, bg="white", selectbackground="blue", selectmode=tk.SINGLE, height=3)
        self.table_body.pack(padx=10, pady=10)

    def open_file(self):
        filename = filedialog.askopenfilename()
        if filename:
            print(f"You chose {filename}")
            self.play_video(filename)

    def object_detection(self):
        self.play_video(r"E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-1.mp4")

    def Trajectory_tracking(self):
        self.play_video(r"E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-track.mp4")

    def Vehicle_Counting(self):
        self.play_video(r"E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-count.mp4")

    def License_plate_detection(self):
        self.play_video(r"E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-code.mp4")

    def Speed_detection(self):
        self.play_video(r"E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-speed.mp4")

    def Overspeed_detection(self):
        self.play_video(r"E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-speed-fast.mp4")

    def Lane_line_detection(self):
      pass

    def Construction_area(self):
        self.play_video(r"E:\Autonomous driving\ultralytics-main\deteciton\video\抖音2024918-824647-area.mp4")

    def play_video(self, video_path):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        canvas_width = self.image_display_area.winfo_width()
        canvas_height = self.image_display_area.winfo_height()

        if self.play_task is not None:
            self.master.after_cancel(self.play_task)

        self.play_task = self.master.after(0, self.update_frame, canvas_width, canvas_height)

    def update_frame(self, canvas_width, canvas_height):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame.")
            self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)  # 重置到视频开头
            ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.resize_frame(frame, (canvas_width, canvas_height))

            # 将 NumPy 数组转换为 PIL.Image 对象
            pil_image = Image.fromarray(frame)
            photo_image = ImageTk.PhotoImage(image=pil_image)

            # 计算居中位置
            frame_width, frame_height = pil_image.size  # 使用 PIL.Image.size
            x = (canvas_width - frame_width) // 2
            y = (canvas_height - frame_height) // 2

            if self.image_id is None:
                self.image_id = self.image_display_area.create_image(x, y, anchor=tk.NW, image=photo_image)
            else:
                self.image_display_area.itemconfig(self.image_id, image=photo_image)
                self.image_display_area.coords(self.image_id, x, y)  # 不需要额外的偏移

            self.image_display_area.image = photo_image
            self.master.update_idletasks()

            # 延迟更长时间以减慢播放速度
            self.play_task = self.master.after(10, self.update_frame, canvas_width, canvas_height)

    def resize_frame(self, frame, target_size):
        """
        Resize the frame to fit into the target size while preserving aspect ratio.
        """
        height, width, _ = frame.shape
        target_width, target_height = target_size
        ratio = min(target_width / width, target_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return resized_frame

    def on_resize(self, event):
        if self.image_id is not None:
            canvas_width = event.width
            canvas_height = event.height
            self.update_frame(canvas_width, canvas_height)

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()