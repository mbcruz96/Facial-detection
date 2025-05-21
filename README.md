# Comparative Study of Facial Detection Algorithms
- **Project Goal:** To comprehensively evaluate and compare various pretrained face detection models, guiding developers and researchers in informed model selection for diverse applications.

- **Models Evaluated:** The study analyzed Haar cascade, dlib CNN, MTCNN, dlib HOG, RetinaFace, and OpenCV CNN models, representing a spectrum of approaches from traditional feature extraction to deep learning.

- **Datasets Used:** Evaluation was conducted using images from the Labelled Faces in the Wild (LFW) dataset and other image datasets, along with video data from the Condensed Movies Dataset to simulate real-world conditions.

**Key Findings:**
- **High Accuracy (but resource intensive):** Dlib CNN (90-95% accuracy) and RetinaFace (85-95% accuracy, sometimes up to 99%) offer high precision but require significant computational resources, making them less suitable for real-time processing.
- **Fast (but lower accuracy):** Haar Cascade (61% accuracy, 98.7% PPV) and HOG (80-90% accuracy) provide rapid processing and efficiency, ideal for real-time applications with limited computational capacity. Haar Cascade is limited to front-facing faces.
- **Balanced Performance:** MTCNN and OpenCV DNN offer a balance of accuracy and efficiency, though MTCNN has high computational cost and is not suitable for real-time processing. OpenCV DNN can have false negatives in videos.
