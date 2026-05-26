import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import serial
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🔥 Fire Detection AI",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.result {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    margin-top: 15px;
    color: white;
}

.fire {
    background-color: #b02a37;
}

.smoke {
    background-color: #997404;
}

.safe {
    background-color: #146c43;
}

.blocked {
    background-color: #0d6efd;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🔥 Fire Detection AI")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_ai_model():

    return load_model(
        "keras_Model.h5",
        compile=False
    )

model = load_ai_model()

# =========================
# LOAD LABELS
# =========================
@st.cache_data
def load_labels():

    return open(
        "labels.txt",
        "r"
    ).readlines()

class_names = load_labels()

# =========================
# AI FUNCTION
# =========================
def predict_fire(img):

    image = cv2.resize(
        img,
        (224, 224)
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ).reshape(1, 224, 224, 3)

    image = (image / 127.5) - 1

    prediction = model.predict(
        image,
        verbose=0
    )

    index = np.argmax(
        prediction
    )

    class_name = class_names[
        index
    ][2:].strip().lower()

    confidence = float(
        prediction[0][index]
    )

    return class_name, confidence

# =========================
# SIDEBAR
# =========================
mode = st.sidebar.selectbox(
    "Mode",
    [
        "Upload Image",
        "Live Detection"
    ]
)

# =========================
# IMAGE MODE
# =========================
if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(
            image
        )

        class_name, confidence = predict_fire(
            image_np
        )

        confidence_percent = int(
            confidence * 100
        )

        st.image(
            image_np,
            width=500
        )

        # FIRE
        if (
            "fire" in class_name
            and confidence > 0.90
        ):

            st.markdown(
                f"""
                <div class="result fire">
                🔥 FIRE {confidence_percent}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # SMOKE
        elif (
            "smoke" in class_name
            and confidence > 0.90
        ):

            st.markdown(
                f"""
                <div class="result smoke">
                💨 SMOKE {confidence_percent}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # BLOCKED
        elif (
            "blocked" in class_name
            and confidence > 0.90
        ):

            st.markdown(
                f"""
                <div class="result blocked">
                🚫 BLOCKED {confidence_percent}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # SAFE
        else:

            st.markdown(
                f"""
                <div class="result safe">
                ✅ SAFE {confidence_percent}%
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================
# LIVE DETECTION
# =========================
elif mode == "Live Detection":

    st.subheader(
        "📷 Phone Camera + 🔥 Flame Sensor"
    )

    # =========================
    # CAMERA URL
    # =========================
    camera_source = st.text_input(
        "Phone Camera URL",
        "http://10.40.21.122:8080/video"
    )

    # =========================
    # COM PORT
    # =========================
    sensor_port = st.text_input(
        "Arduino COM Port",
        "COM5"
    )

    # =========================
    # START
    # =========================
    run = st.checkbox(
        "Start Detection"
    )

    FRAME_WINDOW = st.image([])

    sensor_box = st.empty()

    if run:

        # =========================
        # CAMERA
        # =========================
        camera = cv2.VideoCapture(
            camera_source
        )

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        # =========================
        # SENSOR
        # =========================
        sensor_connected = False

        try:

            arduino = serial.Serial(
                sensor_port,
                9600,
                timeout=1
            )

            time.sleep(2)

            sensor_connected = True

            st.success(
                "✅ Sensor Connected"
            )

        except Exception as e:

            st.warning(
                f"⚠️ Sensor Error: {e}"
            )

        # =========================
        # CAMERA CHECK
        # =========================
        if not camera.isOpened():

            st.error(
                "❌ Camera Not Connected"
            )

        else:

            while run:

                success, frame = camera.read()

                if not success:

                    st.error(
                        "❌ Camera Frame Error"
                    )

                    break

                # Mirror View
                frame = cv2.flip(
                    frame,
                    1
                )

                # =========================
                # AI DETECTION
                # =========================
                try:

                    class_name, confidence = predict_fire(
                        frame
                    )

                    confidence_percent = int(
                        confidence * 100
                    )

                    # FIRE
                    if (
                        "fire" in class_name
                        and confidence > 0.90
                    ):

                        text = (
                            f"🔥 FIRE {confidence_percent}%"
                        )

                        color = (
                            0,
                            0,
                            255
                        )

                    # SMOKE
                    elif (
                        "smoke" in class_name
                        and confidence > 0.90
                    ):

                        text = (
                            f"💨 SMOKE {confidence_percent}%"
                        )

                        color = (
                            0,
                            255,
                            255
                        )

                    # BLOCKED
                    elif (
                        "blocked" in class_name
                        and confidence > 0.90
                    ):

                        text = (
                            f"🚫 BLOCKED {confidence_percent}%"
                        )

                        color = (
                            255,
                            0,
                            0
                        )

                    # SAFE
                    else:

                        text = (
                            f"✅ SAFE {confidence_percent}%"
                        )

                        color = (
                            0,
                            255,
                            0
                        )

                except Exception as e:

                    text = "AI ERROR"

                    color = (
                        120,
                        120,
                        120
                    )

                    print(e)

                # =========================
                # SENSOR DETECTION
                # =========================
                if sensor_connected:

                    try:

                        if arduino.in_waiting:

                            data = arduino.readline().decode().strip()

                            # FIRE
                            if data == "FIRE":

                                sensor_box.markdown(
                                    """
                                    <div class="result fire">
                                    🔥 SENSOR DETECTED FIRE
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            # SAFE
                            elif data == "SAFE":

                                sensor_box.markdown(
                                    """
                                    <div class="result safe">
                                    ✅ SENSOR SAFE
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    except Exception as e:

                        sensor_box.markdown(
                            f"""
                            <div class="result blocked">
                            ⚠️ SENSOR ERROR
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        print(e)

                # =========================
                # DRAW AI TEXT
                # =========================
                cv2.putText(
                    frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                # =========================
                # RGB
                # =========================
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # =========================
                # SHOW FRAME
                # =========================
                FRAME_WINDOW.image(
                    frame,
                    channels="RGB"
                )

                time.sleep(0.03)

            camera.release()

            if sensor_connected:

                arduino.close()