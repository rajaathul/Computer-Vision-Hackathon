OnGuard Safety Detector


OnGuard Safety Detector is a computer vision-based application designed to enhance workplace safety. Using the advanced YOLOv8 model, this system detects whether individuals in construction sites are wearing essential protective equipment like helmets and safety vests. The application allows users to upload images or videos to generate detection results, classifying individuals into "Safe", "Partially Safe", or "Not Safe" categories.






 Features
- Real-time detection of helmets and vests in images or videos.
- Classifies safety compliance into:
  - Safe: Wearing both helmet and vest.
  - Partially Safe: Wearing either helmet or vest.
  - Not Safe: Wearing neither.
- User-friendly interface for uploading media files.
- Supports image and video processing.
- Built with Flask for a lightweight and scalable backend.




 Installation


 Prerequisites
Make sure you have the following installed:
- Python 3.8 or above
- pip (Python package installer)
- Git


 Setup Instructions


1. Clone the Repository
   
   git clone https://github.com/yourusername/onguard-safety-detector.git
   cd onguard-safety-detector
   


2. Create a Virtual Environment
   
   python -m venv venv
   source venv/bin/activate        For macOS/Linux
   venv\Scripts\activate           For Windows




3. Install Dependencies
   Install the required Python packages:
   
   pip install -r requirements.txt
   


4. Download YOLOv8 Model
   - Download the pre-trained YOLOv8 model weights from [Ultralytics YOLOv8 Repository](https://github.com/ultralytics/ultralytics).
   - Place the downloaded `.pt` file in the `models` directory within the project folder.


5. Start the Flask Application
   
   python app.py
   


6. Access the Application
   - Open your web browser and navigate to `http://127.0.0.1:5000`.






 Usage
1. Open the application in your browser.
2. Upload an image or video file of a construction site.
3. Click Submit to process the file.
4. View the detection results, which will classify individuals as Safe, Partially Safe, or Not Safe with visual annotations.




 Technologies Used
- Flask: Backend framework for handling requests and user interactions.
- YOLOv8: Object detection model for PPE detection.
- OpenCV: For image and video processing.
- HTML/CSS: Frontend for the web interface.






 Future Enhancements
- Add real-time video streaming support.
- Integrate a notification system for non-compliance alerts.
- Expand PPE detection to include gloves, goggles, and other equipment.
- Deploy the application on cloud platforms for scalability.






 Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch:
   
   git checkout -b feature-name
   
3. Commit your changes:
   
   git commit -m "Add a new feature"
   
4. Push to the branch:
   
   git push origin feature-name
   
5. Open a Pull Request.






 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.






 Contact
For any questions or feedback, please contact:  
Your Name  
Email: [your-email@example.com]  
GitHub: [yourusername](https://github.com/yourusername)






Feel free to modify this README to match your exact project details, including your name, GitHub repository link, and any additional features or future plans.