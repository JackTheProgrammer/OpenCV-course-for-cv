import cv2
from deepface_antispoofing import DeepFaceAntiSpoofing

# Initialize the anti-spoofing analyzer
analyzer = DeepFaceAntiSpoofing()

# Open the default webcam (index 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze the live frame to check if the face is real
    result = analyzer.analyze_deepface(frame)

    # Extract results
    is_real = result.get("is_real")
    spoof_type = result.get("spoof_type", "Unknown")

    # Set bounding box color (Green for Real, Red for Spoof/Fake)
    color = (0, 255, 0) if is_real == "True" else (0, 0, 255)
    
    # Display the result on the frame
    text = f"Status: {spoof_type}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the webcam feed
    cv2.imshow("Face Liveness Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()