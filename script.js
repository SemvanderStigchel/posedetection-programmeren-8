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
const learnButton = document.getElementById('learnButton');
const accuracytButton = document.getElementById('accuracyButton');
const trainButton = document.getElementById('trainButton');
const trainOption = document.getElementById('learnOption');
const saveButton = document.getElementById('saveModelButton');
let trainLabel;
let localstorageData = [];
let testData = [];
localstorageData = JSON.parse(localStorage.getItem('poses'));
testData = JSON.parse(localStorage.getItem('testdata'));
trainOption.addEventListener('change', () => {
    trainLabel = document.getElementById('learnOption').value;
});


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
        learnButton.addEventListener('click', predictWebcam);
        trainButton.addEventListener('click', train);
        accuracytButton.addEventListener('click', testAccuracy);
        saveButton.addEventListener('click', saveModel);
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
            learn(symbol);
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

function learn(data) {
    console.log("Learn method");
    if (!trainLabel) {
        console.log('No symbol selected');
        return
    }
    // testData.push(
    //     {pose: data, label: trainLabel}
    // );
    // localStorage.setItem('testdata', JSON.stringify(testData));
    localstorageData.push(
        {pose: data, label: trainLabel}
    );
    localstorageData = localstorageData.toSorted(() => (Math.random() - 0.5))
    localStorage.setItem('poses', JSON.stringify(localstorageData));
    nn.addData(data, {label: trainLabel});
    console.log('data added');
}

function train() {
    console.log("Train method");
    const addPoseData = JSON.parse(localStorage.getItem('poses'));
    for (let pose of addPoseData) {
        nn.addData(pose.pose, {label: pose.label});
        console.log(`Add data`);
    }

    nn.normalizeData();
    nn.train({epochs: 30}, () => finishedTraining());
}

function finishedTraining() {
    console.log('Training klaar');
}

function saveModel () {
    nn.save("model", () => console.log("model was saved!"));
}

async function testAccuracy () {
    let goodPrediction = 0;
    let totalPredictions = 0;
    for (let pose of testData) {
        let prediction = await nn.classify(pose.pose);
        totalPredictions++;
        if (prediction[0].label === pose.label) {
            // console.log(`Testlabel komt overeen met voorspelling!`);
            goodPrediction++;
        } else {
            console.log(`Foute voorspelling! Voorspelling: ${prediction[0].label}, testlabel: ${pose.label}`);
        }
    }
    console.log(goodPrediction / totalPredictions * 100 + '% accuracy');
}