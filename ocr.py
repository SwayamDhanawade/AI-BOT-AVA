import easyocr
import pyttsx3
import cv2
import os

def recognize_and_speak_from_camera():
    # Initialize camera capture
    cap = cv2.VideoCapture(0)  # 0 for default camera

    if not cap.isOpened():
        print("❌ Error: Could not access the camera.")
        return
    
    print("📷 Press 'q' to capture the image and read text.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture image.")
            break
        
        # Display the camera feed in a window
        cv2.imshow('Camera Feed', frame)
        
        # Press 'q' to capture and process the image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)  # Save the captured image
            print(f"✅ Image captured: {image_path}")
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

    # Process the captured image with EasyOCR
    try:
        print("🔍 Recognizing text...")
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image_path)
        
        if not results:
            print("⚠️ No text detected. Try again.")
            return

        # Extract recognized text
        recognized_text = ' '.join([res[1] for res in results])
        print("📝 Recognized Text:", recognized_text)

    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return

    # Initialize TTS engine
    try:
        engine = pyttsx3.init()

        # Set voice properties
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 1)  # Volume level

        # Speak the recognized text
        print("🔊 Speaking the text...")
        engine.say(recognized_text)
        engine.runAndWait()

    except Exception as e:
        print(f"❌ TTS Error: {e}")

if __name__ == "__main__":
    recognize_and_speak_from_camera()
