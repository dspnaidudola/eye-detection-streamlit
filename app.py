import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av


st.set_page_config(
    page_title="Haar Cascade Eye Detection",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Haar Cascade Eye Detection")
st.write(
    "Eye detection using OpenCV Haar Cascade. "
    "No deep-learning model is used."
)


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)


if FACE_CASCADE.empty():
    st.error("Face Haar Cascade could not be loaded.")
    st.stop()

if EYE_CASCADE.empty():
    st.error("Eye Haar Cascade could not be loaded.")
    st.stop()


def detect_eyes(frame):

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces internally
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(60, 60)
    )

    output = frame.copy()

    total_eyes = 0

    # Process every detected face
    for (x, y, w, h) in faces:

        face_gray = gray[
            y:y + h,
            x:x + w
        ]

        # Upper 65% of face
        # where eyes are normally located
        upper_face = face_gray[
            :int(h * 0.65),
            :
        ]

        # Detect eyes
        eyes = EYE_CASCADE.detectMultiScale(
            upper_face,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(12, 12)
        )

        # Draw ONLY eye boxes
        for (ex, ey, ew, eh) in eyes:

            # Make box slightly smaller
            mx = int(ew * 0.15)
            my = int(eh * 0.15)

            x1 = x + ex + mx
            y1 = y + ey + my

            x2 = x + ex + ew - mx
            y2 = y + ey + eh - my

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            total_eyes += 1

    return output, total_eyes


class EyeDetectionProcessor(VideoProcessorBase):

    def recv(self, frame):

        # Convert webcam frame
        img = frame.to_ndarray(
            format="bgr24"
        )

        # Detect eyes
        processed, count = detect_eyes(img)

        # Display eye count
        cv2.putText(
            processed,
            f"Eyes detected: {count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            processed,
            format="bgr24"
        )
    
tab1, tab2 = st.tabs(
    [
        "📷 Image Detection",
        "🎥 Live Webcam"
    ]
)

with tab1:

    st.header("📷 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        # Read uploaded image
        pil_image = Image.open(
            uploaded_file
        ).convert("RGB")

        image = np.array(
            pil_image
        )

        # Convert RGB → BGR
        frame = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # Detect eyes
        result, count = detect_eyes(
            frame
        )

        # Convert BGR → RGB
        result_rgb = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns(2)

        # Original image
        with col1:

            st.subheader(
                "Original Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        # Result
        with col2:

            st.subheader(
                "Eye Detection Result"
            )

            st.image(
                result_rgb,
                use_container_width=True
            )

        # Result information
        if count == 0:

            st.warning(
                "No eyes detected. "
                "Try a clear front-facing image."
            )

        else:

            st.success(
                f"{count} eye(s) detected."
            )

with tab2:

    st.header(
        "🎥 Real-Time Webcam Eye Detection"
    )

    st.info(
        "Click START and allow camera permission "
        "when your browser asks."
    )

    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )

    # Webcam
    webrtc_streamer(
        key="eye-detection",

        video_processor_factory=
        EyeDetectionProcessor,

        rtc_configuration=
        RTC_CONFIGURATION,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )

st.divider()

st.caption(
    "Model: OpenCV Haar Cascade | "
    "Face + Eye Classifiers | "
    "Output: Eye Bounding Boxes Only"
)