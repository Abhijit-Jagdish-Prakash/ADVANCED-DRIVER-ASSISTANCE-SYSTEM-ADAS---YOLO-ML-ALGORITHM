

ADAS (Advanced Driver Assistance System)

A real-time Computer Vision pipeline designed for **Lane Detection** and **Forward Collision Warning (FCW)**. This system is specifically optimized to maintain accuracy during adverse weather conditions like rain, fog, and glare using advanced image preprocessing.

## 🚀 Key Features

* **Weather-Resistant Preprocessing:** Uses the **Lab Color Space** and **CLAHE** to see through fog and heavy rain where standard Grayscale/RGB pipelines fail.
* **Intelligent Denoising:** Implements **Morphological Opening** and **Bilateral Filtering** to eliminate "noise" caused by rain streaks and road spray.
* **Object Detection:** Leverages **YOLOv8** for high-speed vehicle detection.
* **Forward Collision Warning:** Dynamically calculates distance to vehicles within the current lane and triggers a visual alert if the safety threshold is breached.
* **Temporal Smoothing:** Uses a weighted alpha-filter to prevent lane "jitter" and maintain a stable driving path.

---

## 🛠 Tech Stack

| Component | Technology |
| --- | --- |
| **Vision Library** | OpenCV (cv2) |
| **Object Detection** | Ultralytics YOLOv8 |
| **Numerical Processing** | NumPy |
| **Language** | Python 3.x |

---

## 📸 How It Works

### 1. The Preprocessing Pipeline

To handle low-visibility environments, the image undergoes the following transformation:

1. **Lab Color Space Conversion:** Isolate the $L$ (Lightness) channel to handle varying illumination.
2. **CLAHE:** Adaptive histogram equalization to enhance contrast in foggy areas.
3. **Morphological Opening:** Removes small white artifacts (rain drops).
4. **Bilateral Filter:** Blurs noise while preserving the sharp edges of lane markings.

### 2. Distance Estimation

The system estimates distance ($d$) using the pinhole camera model based on the bounding box width of detected vehicles:


$$d = \frac{f \cdot W_{real}}{W_{pixels}}$$

---

## 🚦 Getting Started

### Prerequisites

* Python 3.8 to 3.10 (Higher versions may not support ML)
* A video file named `car.mp4` (There are several other files added in the folder. To change the video file change the name of the file in the code to (video/NAME OF THE FILE)) inside a `video/` directory.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Weather-Resistant-ADAS.git
cd Weather-Resistant-ADAS

```


2. Install dependencies:
```bash
pip install opencv-python numpy ultralytics

```



### Running the System

```bash
python main.py

```

---

## 📂 Project Structure

* `main.py`: The core pipeline containing image preprocessing, lane fitting, and YOLO integration.
* `yolov8n.pt`: (Auto-downloaded) The lightweight YOLOv8 Nano model for real-time inference.
* `video/`: Directory for input footage.

---

## 📝 License

Distributed under the GNU License. See `LICENSE` for more information.
