from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from model_inference import yolo_inference, get_available_models
import uuid
from typing import Optional


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SECRET_KEY'] = 'cairocoders-ednalan'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'webp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/infer', methods=['POST'])
def infer():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Get optional parameters
            model_name = request.form.get('model', 'yolo11n.pt')
            conf_threshold = float(request.form.get('confidence', 0.25))
            iou_threshold = float(request.form.get('iou', 0.45))

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            saveLocation = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Generate unique filename for output image
            unique_filename = str(uuid.uuid4()) + '_' + filename
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], unique_filename)

            try:
                # Update model if different from current one
                if model_name != yolo_inference.model_path:
                    yolo_inference.change_model(model_name)

                # Update confidence threshold if different
                original_conf = yolo_inference.conf_threshold
                if conf_threshold != original_conf:
                    yolo_inference.conf_threshold = conf_threshold

                # Update IOU threshold if different
                original_iou = yolo_inference.iou_threshold
                if iou_threshold != original_iou:
                    yolo_inference.iou_threshold = iou_threshold

                # Perform object detection using the model_inference module
                result = yolo_inference.detect(
                    image_path=saveLocation,
                    output_path=output_image_path
                )

                # Restore original thresholds
                if conf_threshold != original_conf:
                    yolo_inference.conf_threshold = original_conf
                if iou_threshold != original_iou:
                    yolo_inference.iou_threshold = original_iou

                return render_template(
                    'inference.html',
                    saveLocation=saveLocation,
                    output_image=output_image_path,
                    result=result
                )
            except Exception as e:
                flash(f"Error during detection: {str(e)}", 'error')
                return redirect(url_for('home'))

    flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'error')
    return redirect(url_for('home'))


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for object detection"""
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return {'error': 'Invalid file type. Only PNG, JPG, JPEG, and WEBP are allowed.'}, 400

    # Get optional parameters
    model_name = request.form.get('model', 'yolo11n.pt')
    conf_threshold = float(request.form.get('confidence', 0.25))
    iou_threshold = float(request.form.get('iou', 0.45))

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    saveLocation = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Generate unique filename for output image
    unique_filename = str(uuid.uuid4()) + '_' + filename
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], unique_filename)

    try:
        # Update model if different from current one
        if model_name != yolo_inference.model_path:
            yolo_inference.change_model(model_name)

        # Update confidence threshold if different
        original_conf = yolo_inference.conf_threshold
        if conf_threshold != original_conf:
            yolo_inference.conf_threshold = conf_threshold

        # Update IOU threshold if different
        original_iou = yolo_inference.iou_threshold
        if iou_threshold != original_iou:
            yolo_inference.iou_threshold = iou_threshold

        result = yolo_inference.detect(
            image_path=saveLocation,
            output_path=output_image_path
        )

        # Restore original thresholds
        if conf_threshold != original_conf:
            yolo_inference.conf_threshold = original_conf
        if iou_threshold != original_iou:
            yolo_inference.iou_threshold = original_iou

        return {
            'success': True,
            'result': result,
            'original_image': saveLocation,
            'output_image': output_image_path
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}, 500


@app.route('/api/models', methods=['GET'])
def api_models():
    """API endpoint to get available models"""
    models_data = get_available_models()
    return {
        'predefined_models': models_data['predefined_models'],
        'custom_models': models_data['custom_models'],
        'default_model': 'yolo11n.pt'
    }


if __name__ == "__main__":
    # Create upload and output directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    app.run(debug=True)