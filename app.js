// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import {HandLandmarker, FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const nn = ml5.neuralNetwork({task: 'classification', debug: true});
const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
}
nn.load(modelDetails, () => console.log("het model is geladen!"));
const predictButton = document.getElementById('predictButton');


const demosSection = document.getElementById("demos");
let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 1
    });
};
createHandLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
// const drawingUtils = new DrawingUtils(canvasCtx);
// Check if webcam access is supported.
const hasGetUserMedia = () => {
    let _a;
    return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia);
};
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "Enable Spell Casting";
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "Disable Spell Casting";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.onloadedmetadata = async () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
        }


        // video.addEventListener("loadeddata", predictWebcam);
        predictButton.addEventListener('click', predictWebcam);
    });
}

let lastVideoTime = -1;
let results = undefined;
let symbol = [];
let drawingFrames = 0;
console.log(video);

async function predictWebcam() {

    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({runningMode: "VIDEO"});
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }
    // canvasCtx.save();


    if (results.landmarks[0]) {
        // for (const landmarks of results.landmarks) {
        // drawConnectors(canvasCtx, results.landmarks[0], HAND_CONNECTIONS, {
        //     color: "#00FF00",
        //     lineWidth: 5
        // });
        // drawLandmarks(canvasCtx, results.landmarks[0], {color: "#FF0000", lineWidth: 2});
        const pointerFinger = [results.landmarks[0][8]];
        const mapLm = pointerFinger.map(point => [point.x, point.y]);
        const lm = mapLm.flat();
        symbol = symbol.concat(lm);
        canvasCtx.beginPath();
        canvasCtx.fillStyle = "blue";
        canvasCtx.strokeStyle = "blue";
        canvasCtx.lineWidth = 4;
        canvasCtx.arc(canvasElement.width * pointerFinger[0].x, canvasElement.height * pointerFinger[0].y, 12, 0, 2 * Math.PI);
        canvasCtx.fill();
        canvasCtx.closePath();
        // }
        if (drawingFrames++ > 240) {
            console.log('stop');
            drawingFrames = 0;
            await predict(symbol);
            symbol = [];
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            return
        }
    }
    // canvasCtx.restore();


    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

async function predict(data) {
    console.log("Predict method");
    let prediction = await nn.classify(data);
    console.log(prediction);
    console.log(`My prediction is: ${prediction[0].label}`);
    const performedSpell = document.getElementById('performedSpell');
    if (prediction[0].label === 'historia-viventem') {
        prediction[0].label = 'historia viventem';
    }
    performedSpell.innerHTML = `You performed ${prediction[0].label}!`;
    await spellInfo(prediction[0].label);
}

async function spellInfo(spell) {
    const response = await fetch(`http://localhost:3000/spell`, {
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "query": spell,
        })
    });

    const answer = await response.json();
    console.log(answer);
    const displayAnswer = document.getElementById('generated-content');
    displayAnswer.innerHTML = answer;
}