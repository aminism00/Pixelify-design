# Pixelify — AI Pixel Remapping (Java & Python)

Pixelify is a simple AI-based pixel–remapping tool that attempts to reconstruct
a given source image using the color/structure distribution of another target image.
It works by computing a cost matrix between all pixel pairs and solving a minimal–
matching problem (greedy approximation for performance).

This project contains **two implementations**:

- ✅ Java implementation (full pixel–pair cost computation)
- ✅ Python implementation (Hungarian assignment / greedy fallback)

---

## 🎯 Features

- Full–resolution processing (no cropping)
- Output image is exactly the same size as the source image
- Optional spatial proximity weighting
- Works on any pair of images
- Pure Java / Python (no external AI models)

---

## 📁 Project Structure


/java/Pixelify.java → Full high-resolution version
/python/Pixelify.py → Python version
/examples/ → Example input/output images

yaml
Copy code

---

# 🚀 Java Version

### **Compile**

```sh
javac Pixelify.java

java Pixelify source.jpg target.jpg output.png [proximity]
• proximity is optional (default = 0.5)
• Larger proximity → output becomes more structurally similar to target image
• Lower proximity → output preserves more color similarity

🐍 Python Version
Install dependencies
sh
Copy code
pip install pillow numpy scipy
Run
sh
Copy code
python Pixelify.py source.jpg target.jpg 
📷 Examples
powershell
Copy code
source.jpg   →   rebuild using target.jpg   →   output.png
Place your example images in /examples/.

📜 License
MIT License — free to use, modify, and distribute.

🤝 Contributing
Pull requests are welcome.
For major changes, please open an issue first to discuss your ideas.

⭐ If you find this project useful, consider giving it a star!
