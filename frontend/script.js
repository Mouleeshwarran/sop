// =========================================================
// R.A.H.A.T - REAL-TIME ASSISTIVE NAVIGATION
// =========================================================

const video = document.getElementById("camera");
const placeholder = document.getElementById("camera-placeholder");

const canvas = document.getElementById("detectionCanvas");
const ctx = canvas.getContext("2d");
const pathCanvas =
    document.getElementById("forwardPathCanvas");

const pathCtx =
    pathCanvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const switchBtn = document.getElementById("switchBtn");

const voiceText = document.getElementById("voiceText");
const alertTime = document.getElementById("alertTime");

const forwardPath = document.getElementById("forwardPath");

let cameraStream = null;
let cameras = [];
let currentCameraIndex = 0;

let detectionRunning = false;
let detectionInProgress = false;

let latestDetections = [];

let detectionFPS = 0;
let fpsFrames = 0;
let fpsStartTime = performance.now();

let lastInstruction = "";
let lastSpokenInstruction = "";
let lastSpokenTime = 0;

const VOICE_COOLDOWN = 2500;
const DETECTION_INTERVAL = 120;


// =========================================================
// START CAMERA
// =========================================================

async function startCamera() {

    try {

        cameraStream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "environment",
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

        video.srcObject = cameraStream;

        video.style.display = "block";
        placeholder.style.display = "none";

        await video.play();

        await getCameras();

        detectionRunning = true;
        latestDetections = [];

        voiceText.textContent =
            "Navigation started. AI detection is active.";

        alertTime.textContent =
            new Date().toLocaleTimeString();

        requestAnimationFrame(renderLoop);

        detectionLoop();

    } catch (error) {

        console.error("Camera error:", error);

        voiceText.textContent =
            "Camera access denied or unavailable.";

        alertTime.textContent =
            new Date().toLocaleTimeString();

        alert(
            "Please allow camera access in your browser."
        );
    }
}


// =========================================================
// STOP CAMERA
// =========================================================

function stopCamera() {

    detectionRunning = false;
    detectionInProgress = false;

    latestDetections = [];

    clearCanvas();

    if (cameraStream) {

        cameraStream.getTracks().forEach(
            track => track.stop()
        );

        cameraStream = null;
    }

    video.srcObject = null;

    video.style.display = "none";
    placeholder.style.display = "flex";

    voiceText.textContent =
        "Navigation stopped.";

    alertTime.textContent =
        new Date().toLocaleTimeString();

    resetDashboard();
}


// =========================================================
// GET CAMERAS
// =========================================================

async function getCameras() {

    try {

        const devices =
            await navigator.mediaDevices.enumerateDevices();

        cameras =
            devices.filter(
                device =>
                    device.kind === "videoinput"
            );

        console.log(
            "Available cameras:",
            cameras
        );

    } catch (error) {

        console.error(
            "Camera enumeration error:",
            error
        );
    }
}


// =========================================================
// SWITCH CAMERA
// =========================================================

async function switchCamera() {

    if (cameras.length <= 1) {

        alert(
            "Only one camera is available."
        );

        return;
    }

    currentCameraIndex =
        (currentCameraIndex + 1) %
        cameras.length;

    const cameraId =
        cameras[currentCameraIndex].deviceId;


    if (cameraStream) {

        cameraStream.getTracks().forEach(
            track => track.stop()
        );
    }


    try {

        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    deviceId: {
                        exact: cameraId
                    },

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    }
                },

                audio: false
            });

        video.srcObject =
            cameraStream;

        await video.play();

        voiceText.textContent =
            "Camera switched.";

        alertTime.textContent =
            new Date().toLocaleTimeString();

    } catch (error) {

        console.error(
            "Camera switch error:",
            error
        );
    }
}


// =========================================================
// YOLO DETECTION LOOP
// =========================================================

async function detectionLoop() {

    if (!detectionRunning) {
        return;
    }


    if (
        !detectionInProgress &&
        video.readyState >= 2 &&
        video.videoWidth > 0
    ) {

        detectionInProgress = true;

        try {

            const result =
                await sendFrameToYOLO();


            if (
                result &&
                result.success
            ) {

                latestDetections =
                    result.detections || [];


                updateDashboard(
                    latestDetections,
                    result.navigation
                );


                updateFPS();
            }

        } catch (error) {

            console.error(
                "Detection error:",
                error
            );

        } finally {

            detectionInProgress = false;
        }
    }


    if (detectionRunning) {

        setTimeout(
            detectionLoop,
            DETECTION_INTERVAL
        );
    }
}


// =========================================================
// SEND FRAME TO BACKEND
// =========================================================

async function sendFrameToYOLO() {

    const captureCanvas =
        document.createElement("canvas");


    captureCanvas.width =
        video.videoWidth;

    captureCanvas.height =
        video.videoHeight;


    const captureContext =
        captureCanvas.getContext("2d");


    captureContext.drawImage(
        video,
        0,
        0,
        video.videoWidth,
        video.videoHeight
    );


    const blob =
        await new Promise(resolve => {

            captureCanvas.toBlob(
                resolve,
                "image/jpeg",
                0.65
            );

        });


    if (!blob) {
        return null;
    }


    const formData =
        new FormData();


    formData.append(
        "file",
        blob,
        "camera.jpg"
    );


    const response =
        await fetch(
            "http://127.0.0.1:8000/detect",
            {
                method: "POST",
                body: formData
            }
        );


    if (!response.ok) {

        throw new Error(
            `Backend error: ${response.status}`
        );
    }


    return await response.json();
}


// =========================================================
// SMOOTH RENDER LOOP
// =========================================================

function renderLoop() {

    if (!detectionRunning) {

        clearCanvas();
        clearPathCanvas();

        return;
    }

    drawForwardPath();
    drawDetections();

    requestAnimationFrame(
        renderLoop
    );
}

// =========================================================
// DRAW DETECTIONS
// =========================================================

function drawDetections() {

    const displayWidth =
        canvas.clientWidth;

    const displayHeight =
        canvas.clientHeight;


    if (
        displayWidth === 0 ||
        displayHeight === 0
    ) {
        return;
    }


    canvas.width =
        displayWidth;

    canvas.height =
        displayHeight;


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    const videoWidth =
        video.videoWidth;

    const videoHeight =
        video.videoHeight;


    if (
        videoWidth === 0 ||
        videoHeight === 0
    ) {
        return;
    }


    // Account for object-fit: cover
    const scale =
        Math.max(
            displayWidth / videoWidth,
            displayHeight / videoHeight
        );


    const renderedWidth =
        videoWidth * scale;

    const renderedHeight =
        videoHeight * scale;


    const offsetX =
        (displayWidth - renderedWidth) / 2;

    const offsetY =
        (displayHeight - renderedHeight) / 2;


    latestDetections.forEach(
        detection => {

            const x1 =
                detection.x1 *
                scale +
                offsetX;


            const y1 =
                detection.y1 *
                scale +
                offsetY;


            const x2 =
                detection.x2 *
                scale +
                offsetX;


            const y2 =
                detection.y2 *
                scale +
                offsetY;


            const width =
                x2 - x1;

            const height =
                y2 - y1;


            drawDetectionBox(
                x1,
                y1,
                width,
                height,
                detection
            );
        }
    );
}


// =========================================================
// DRAW ONE OBJECT
// =========================================================

function drawDetectionBox(
    x,
    y,
    width,
    height,
    detection
) {

    // RED = object is directly in forward path
    //
    // YELLOW = side object / outside immediate path

    const color =
        detection.in_forward_path
            ? "#ff3030"
            : "#ffd43b";


    ctx.strokeStyle =
        color;

    ctx.lineWidth =
        detection.in_forward_path
            ? 5
            : 3;


    ctx.strokeRect(
        x,
        y,
        width,
        height
    );


    const confidence =
        Math.round(
            detection.confidence * 100
        );


    const label =
        `${formatClassName(detection.class)} ${confidence}%`;


    ctx.font =
        "bold 16px Arial";


    const textWidth =
        ctx.measureText(label).width;


    const labelHeight =
        27;


    ctx.fillStyle =
        color;


    ctx.fillRect(
        x,
        Math.max(
            0,
            y - labelHeight
        ),
        textWidth + 14,
        labelHeight
    );


    ctx.fillStyle =
        "#000000";


    ctx.fillText(
        label,
        x + 7,
        Math.max(
            18,
            y - 8
        )
    );


    // -----------------------------------------------------
    // Show distance and path status
    // -----------------------------------------------------

    const info =
        `${detection.distance} • ${detection.direction}`;


    ctx.font =
        "bold 13px Arial";


    const infoWidth =
        ctx.measureText(info).width;


    ctx.fillStyle =
        "rgba(0, 0, 0, 0.75)";


    ctx.fillRect(
        x,
        y + height,
        infoWidth + 12,
        22
    );


    ctx.fillStyle =
        "#ffffff";


    ctx.fillText(
        info,
        x + 6,
        y + height + 16
    );
}


// =========================================================
// UPDATE DASHBOARD
// =========================================================

function updateDashboard(
    detections,
    navigation
) {

    // -----------------------------------------------------
    // Object statistics
    // -----------------------------------------------------

    const potholes =
        detections.filter(
            d => d.type === "pothole"
        ).length;


    const pedestrians =
        detections.filter(
            d => d.class === "person"
        ).length;


    const bicycles =
        detections.filter(
            d => d.class === "bicycle"
        ).length;


    const vehicles =
        detections.filter(
            d =>
                d.class === "car" ||
                d.class === "motorcycle" ||
                d.class === "bus" ||
                d.class === "truck"
        ).length;


    document.getElementById(
        "potholes"
    ).textContent =
        potholes;


    document.getElementById(
        "pedestrians"
    ).textContent =
        pedestrians;


    document.getElementById(
        "bicycles"
    ).textContent =
        bicycles;


    document.getElementById(
        "vehicles"
    ).textContent =
        vehicles;


    document.getElementById(
        "fps"
    ).textContent =
        detectionFPS;


    document.getElementById(
        "summaryFps"
    ).textContent =
        detectionFPS;


    // -----------------------------------------------------
    // IMPORTANT:
    //
    // Navigation decision comes from BACKEND.
    // Frontend does not decide based on object class.
    // -----------------------------------------------------

    if (navigation) {

        updateNavigation(
            navigation
        );

    } else {

        updateClearState();
    }
}


// =========================================================
// UPDATE NAVIGATION UI
// =========================================================

function updateNavigation(
    navigation
) {

    const target =
        navigation.target;


    const warningTitle =
        document.getElementById(
            "warningTitle"
        );


    const warningDistance =
        document.getElementById(
            "warningDistance"
        );


    const warningDirection =
        document.getElementById(
            "warningDirection"
        );


    const warningConfidence =
        document.getElementById(
            "warningConfidence"
        );


    const suggestedAction =
        document.getElementById(
            "suggestedAction"
        );


    const navigationStatus =
        document.getElementById(
            "navigationStatus"
        );


    const navigationMessage =
        document.getElementById(
            "navigationMessage"
        );


    // -----------------------------------------------------
    // PATH CLEAR
    // -----------------------------------------------------

    if (!target) {

        warningTitle.textContent =
            "✓ NO OBSTACLE";


        warningDistance.textContent =
            "--";


        warningDirection.textContent =
            "--";


        warningConfidence.textContent =
            "--";


        suggestedAction.textContent =
            navigation.instruction ||
            "PATH CLEAR";


        navigationStatus.textContent =
            "🛣️ PATH CLEAR";


        navigationMessage.textContent =
            navigation.reason ||
            "Safe to proceed";


        voiceText.textContent =
            "✓ Path clear.";


        alertTime.textContent =
            new Date().toLocaleTimeString();


        return;
    }


    // -----------------------------------------------------
    // TARGET EXISTS
    // -----------------------------------------------------

    warningTitle.textContent =
        "⚠️ OBSTACLE DETECTED";


    warningDistance.textContent =
        target.distance;


    warningDirection.textContent =
        target.direction;


    warningConfidence.textContent =
        `${Math.round(
            target.confidence * 100
        )}%`;


    suggestedAction.textContent =
        navigation.instruction;


    navigationStatus.textContent =
        `⚠️ ${navigation.status}`;


    navigationMessage.textContent =
        navigation.reason;


    // -----------------------------------------------------
    // Voice/message uses the DECISION,
    // not the object category.
    // -----------------------------------------------------

    const instruction =
        navigation.instruction;

if (
    instruction !== lastInstruction
) {

    voiceText.textContent =
        `⚠️ ${instruction}`;

    alertTime.textContent =
        new Date().toLocaleTimeString();

    speakNavigation(
        instruction
    );

    lastInstruction =
        instruction;
}
}


// =========================================================
// RESET
// =========================================================

function resetDashboard() {

    document.getElementById(
        "potholes"
    ).textContent = "0";


    document.getElementById(
        "pedestrians"
    ).textContent = "0";


    document.getElementById(
        "bicycles"
    ).textContent = "0";


    document.getElementById(
        "vehicles"
    ).textContent = "0";


    document.getElementById(
        "fps"
    ).textContent = "0";


    document.getElementById(
        "summaryFps"
    ).textContent = "0";


    updateClearState();
}


// =========================================================
// CLEAR STATE
// =========================================================

function updateClearState() {

    document.getElementById(
        "warningTitle"
    ).textContent =
        "✓ NO OBSTACLE";


    document.getElementById(
        "warningDistance"
    ).textContent =
        "--";


    document.getElementById(
        "warningDirection"
    ).textContent =
        "--";


    document.getElementById(
        "warningConfidence"
    ).textContent =
        "--";


    document.getElementById(
        "suggestedAction"
    ).textContent =
        "PATH CLEAR";


    document.getElementById(
        "navigationStatus"
    ).textContent =
        "🛣️ PATH CLEAR";


    document.getElementById(
        "navigationMessage"
    ).textContent =
        "Safe to proceed";


    voiceText.textContent =
        "✓ Path clear.";
}


// =========================================================
// FPS
// =========================================================

function updateFPS() {

    fpsFrames++;

    const now =
        performance.now();


    const elapsed =
        now - fpsStartTime;


    if (elapsed >= 1000) {

        detectionFPS =
            Math.round(
                fpsFrames * 1000 / elapsed
            );


        fpsFrames = 0;

        fpsStartTime =
            now;
    }
}


// =========================================================
// FORMAT CLASS
// =========================================================

function formatClassName(
    className
) {

    return className
        .charAt(0)
        .toUpperCase()
        + className.slice(1);
}


// =========================================================
// CLEAR CANVAS
// =========================================================

function clearCanvas() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
}


// =========================================================
// BUTTON EVENTS
// =========================================================

startBtn.addEventListener(
    "click",
    startCamera
);

stopBtn.addEventListener(
    "click",
    stopCamera
);

switchBtn.addEventListener(
    "click",
    switchCamera
);

// =========================================================
// DYNAMIC FORWARD PATH
// =========================================================

function drawForwardPath() {

    const width =
        pathCanvas.clientWidth;

    const height =
        pathCanvas.clientHeight;

    if (!width || !height) {
        return;
    }

    pathCanvas.width = width;
    pathCanvas.height = height;

    pathCtx.clearRect(
        0,
        0,
        width,
        height
    );


    // Forward path trapezoid.
    // Narrow far away, wider near the camera.

    const topY =
        height * 0.35;

    const bottomY =
        height;

    const topLeft =
        width * 0.42;

    const topRight =
        width * 0.58;

    const bottomLeft =
        width * 0.10;

    const bottomRight =
        width * 0.90;


    pathCtx.beginPath();

    pathCtx.moveTo(
        topLeft,
        topY
    );

    pathCtx.lineTo(
        topRight,
        topY
    );

    pathCtx.lineTo(
        bottomRight,
        bottomY
    );

    pathCtx.lineTo(
        bottomLeft,
        bottomY
    );

    pathCtx.closePath();


    // Transparent path region

    pathCtx.fillStyle =
        "rgba(40, 220, 90, 0.16)";

    pathCtx.fill();


    // Path boundaries

    pathCtx.strokeStyle =
        "#45ff65";

    pathCtx.lineWidth = 3;

    pathCtx.setLineDash([
        10,
        8
    ]);

    pathCtx.beginPath();

    pathCtx.moveTo(
        topLeft,
        topY
    );

    pathCtx.lineTo(
        bottomLeft,
        bottomY
    );

    pathCtx.moveTo(
        topRight,
        topY
    );

    pathCtx.lineTo(
        bottomRight,
        bottomY
    );

    pathCtx.stroke();

    pathCtx.setLineDash([]);


    // Label

    pathCtx.fillStyle =
        "#ffffff";

    pathCtx.font =
        "bold 17px Arial";

    pathCtx.textAlign =
        "center";

    pathCtx.fillText(
        "FORWARD PATH",
        width / 2,
        topY + 30
    );
}


// =========================================================
// CLEAR PATH CANVAS
// =========================================================

function clearPathCanvas() {

    pathCtx.clearRect(
        0,
        0,
        pathCanvas.width,
        pathCanvas.height
    );
}
// =========================================================
// VOICE NAVIGATION
// =========================================================

function speakNavigation(instruction) {

    const voiceToggle =
        document.getElementById(
            "voiceToggle"
        );

    // Voice alerts disabled
    if (
        voiceToggle &&
        !voiceToggle.checked
    ) {
        return;
    }


    if (
        !("speechSynthesis" in window)
    ) {
        console.warn(
            "Speech synthesis is not supported."
        );

        return;
    }


    const now =
        Date.now();


    // Prevent repeated announcements
    if (
        instruction === lastSpokenInstruction &&
        now - lastSpokenTime <
        VOICE_COOLDOWN
    ) {
        return;
    }


    // Stop previous speech
    window.speechSynthesis.cancel();


    const message =
        createVoiceMessage(
            instruction
        );


    const utterance =
        new SpeechSynthesisUtterance(
            message
        );


    const language =
        document.getElementById(
            "languageSelect"
        );


    if (
        language &&
        language.value === "ta"
    ) {

        utterance.lang =
            "ta-IN";

    } else if (
        language &&
        language.value === "hi"
    ) {

        utterance.lang =
            "hi-IN";

    } else {

        utterance.lang =
            "en-IN";
    }


    utterance.rate =
        0.95;

    utterance.pitch =
        1.0;

    utterance.volume =
        1.0;


    window.speechSynthesis.speak(
        utterance
    );


    lastSpokenInstruction =
        instruction;

    lastSpokenTime =
        now;
}


// =========================================================
// NATURAL VOICE MESSAGE
// =========================================================

function createVoiceMessage(
    instruction
) {

    switch (instruction) {

        case "PATH CLEAR":
            return "Path clear. Safe to proceed.";

        case "SLOW DOWN":
            return "Obstacle ahead. Slow down.";

        case "MOVE LEFT":
            return "Obstacle ahead. Move left.";

        case "MOVE RIGHT":
            return "Obstacle ahead. Move right.";

        case "KEEP LEFT":
            return "Obstacle on the right. Keep left.";

        case "KEEP RIGHT":
            return "Obstacle on the left. Keep right.";

        case "WAIT / STOP":
            return "Obstacle ahead. Wait. Stop.";

        case "PROCEED WITH CAUTION":
            return "Obstacles nearby. Proceed with caution.";

        default:
            return instruction;
    }
}