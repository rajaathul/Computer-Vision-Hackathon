import traceback
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from werkzeug.utils import secure_filename
from both import detect_and_process  # Import the function from both.py
from detect_video_process import detect_and_process_video

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Folder to save the processed file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    # Check if the file is of the allowed types
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Check if file is allowed
        if file and allowed_file(file.filename):
            try:
                file_ext = os.path.splitext(file.filename)[1].lower()
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                
                # Check if it's an image or video
                if file_ext in ['.mp4', '.avi', '.mov']:
                    print("Video Detected")
                    # Process the video
                    output_video_path = app.config['UPLOAD_FOLDER']
                    processed_video_path = detect_and_process_video(file_path,output_video_path)
                    
                    # Redirect to video result page
                    return redirect(url_for('show_video_result', filename=os.path.basename(processed_video_path)))
                else:
                    # If it's an image, process the image
                    processed_file_path, detection_counts = detect_and_process(file_path)

                    # Redirect to the result page with the processed image and detection counts
                    return redirect(url_for('show_result', filename=os.path.basename(processed_file_path),
                                            total_people=detection_counts['total_people'],
                                            num_safe=detection_counts['safe_count'],
                                            num_partially_safe=detection_counts['partially_safe_count'],
                                            num_not_safe=detection_counts['not_safe_count']))
            except Exception as e:
                print("An error occurred during detection:")
                print(traceback.format_exc())
                return render_template('error.html', error_message=str(e))
    return render_template('index.html')

@app.route('/result/<filename>')
def show_result(filename):
    # Get the detection counts from the query parameters
    total_people = request.args.get('total_people', 0, type=int)
    num_safe = request.args.get('num_safe', 0, type=int)
    num_partially_safe = request.args.get('num_partially_safe', 0, type=int)
    num_not_safe = request.args.get('num_not_safe', 0, type=int)

    # Pass the filename and detection counts to the results.html template
    return render_template('results.html', filename=filename,
                           total_people=total_people,
                           num_safe=num_safe,
                           num_partially_safe=num_partially_safe,
                           num_not_safe=num_not_safe)


@app.route('/video_result/<filename>')
def show_video_result(filename):
    return render_template('video_result.html', video_file=filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve the file from the uploads folder
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
