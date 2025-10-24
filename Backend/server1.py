from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import datetime

app = Flask(__name__)
CORS(app)  # <-- allow frontend requests

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(ROOT_FOLDER, "uploaded_history")
os.makedirs(SAVE_FOLDER, exist_ok=True)

@app.route("/save-file", methods=["POST"])
def save_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(SAVE_FOLDER, filename)
    file.save(file_path)
    mtime = os.path.getmtime(file_path)
    return jsonify({
        "success": True,
        "filename": filename,
        "timestamp": mtime,
        "date": datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/list-files", methods=["GET"])
def list_files():
    files = []
    for f in sorted(os.listdir(SAVE_FOLDER), reverse=True):
        path = os.path.join(SAVE_FOLDER, f)
        if os.path.isfile(path):
            mtime = os.path.getmtime(path)
            files.append({
                "filename": f,
                "timestamp": mtime,
                "date": datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
    return jsonify(files)

if __name__ == "__main__":
    print(f"Files will be saved in: {SAVE_FOLDER}")
    app.run(port=8051, debug=True)
