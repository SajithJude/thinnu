# import streamlit as st
# import streamlit.components.v1 as components

# components.html(
#     """
#  <!DOCTYPE html>
# <html>
# <head>
#   <meta charset="utf-8">
#   <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
#   <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
#   <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
#   <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" crossorigin="anonymous"></script>
# </head>

# <body>
#   <div class="container">
#     <video class="input_video"></video>
#     <canvas class="output_canvas" width="700px" height="400px"></canvas>
#   </div>
#   <script type="module">
# const videoElement = document.getElementsByClassName('input_video')[0];
# const canvasElement = document.getElementsByClassName('output_canvas')[0];
# const canvasCtx = canvasElement.getContext('2d');

# function onResults(results) {
#   canvasCtx.save();
#   canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
#   canvasCtx.drawImage(
#       results.image, 0, 0, canvasElement.width, canvasElement.height);
#   if (results.multiFaceLandmarks) {
#     for (const landmarks of results.multiFaceLandmarks) {
#       drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION,
#                      {color: '#C0C0C070', lineWidth: 1});
#       drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {color: '#FF3030'});
#       drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYEBROW, {color: '#FF3030'});
#       drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_IRIS, {color: '#FF3030'});
#       drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {color: '#30FF30'});
#       drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYEBROW, {color: '#30FF30'});
#       drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_IRIS, {color: '#30FF30'});
#       drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {color: '#E0E0E0'});
#       drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, {color: '#E0E0E0'});
#     }
#   }
#   canvasCtx.restore();
# }

# const faceMesh = new FaceMesh({locateFile: (file) => {
#   return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
# }});
# faceMesh.setOptions({
#   maxNumFaces: 1,
#   refineLandmarks: true,
#   minDetectionConfidence: 0.5,
#   minTrackingConfidence: 0.5
# });
# faceMesh.onResults(onResults);

# const camera = new Camera(videoElement, {
#   onFrame: async () => {
#     await faceMesh.send({image: videoElement});
#   },
#   width: 300,
#   height: 200
# });
# camera.start();
# </script>
# </body>

# </html>
#     """,
#     height=600,
# )
