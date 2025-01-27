import cv2
import time
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

# Setup: Get the number of rounds required to win the game
player_1_score = 0
player_2_score = 0
required_rounds = int(input("Enter the number of rounds required to win the game: "))

# Define a function to check if a player has won
def check_winner(player_1_score, player_2_score, required_rounds):
    if player_1_score >= required_rounds:
        print("Player 1 wins the game!")
        return True
    elif player_2_score >= required_rounds:
        print("Player 2 wins the game!")
        return True
    return False

# Function to calculate the movement of a hand between two frames
def has_moved(prev_position, current_position, threshold=20):
    if not prev_position or not current_position:
        return False  # No movement detected if one of the positions is missing
    x1_prev, y1_prev, x2_prev, y2_prev = prev_position
    x1_curr, y1_curr, x2_curr, y2_curr = current_position

    # Calculate the Euclidean distance for bounding box centers
    center_prev = ((x1_prev + x2_prev) // 2, (y1_prev + y2_prev) // 2)
    center_curr = ((x1_curr + x2_curr) // 2, (y1_curr + y2_curr) // 2)

    distance = ((center_curr[0] - center_prev[0]) ** 2 + (center_curr[1] - center_prev[1]) ** 2) ** 0.5
    return distance > threshold

# Wait until both players show "rock" before starting the game
def wait_for_both_players_to_show_rock():
    print("Waiting for both players to show 'rock' to start the game...")
    countdown_started = False
    countdown_start_time = None

    # Initialize previous positions for tracking hand movement
    prev_positions = {"player_1": None, "player_2": None}
    movement_detected = {"player_1": False, "player_2": False}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO prediction
        results = model(frame)

        # Get detections for this frame
        detections = results[0]

        # Define placeholders for player detections
        player_1_rock = False
        player_2_rock = False
        current_positions = {"player_1": None, "player_2": None}

        # Check if detections are present
        if detections.boxes is not None:
            for box in detections.boxes:
                # Extract bounding box coordinates, confidence, and class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])  # Class ID
                class_name = model.names[cls]  # Get the name of the class

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for waiting phase
                label = f"{class_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Determine which player the object belongs to based on x-coordinate
                center_x = (x1 + x2) // 2
                if class_name.lower() == "rock":
                    if center_x < frame_width // 2:
                        player_1_rock = True
                        current_positions["player_1"] = (x1, y1, x2, y2)
                    else:
                        player_2_rock = True
                        current_positions["player_2"] = (x1, y1, x2, y2)

        # Check if both players show "rock"
        if player_1_rock and player_2_rock and not countdown_started:
            print("Both players showed 'rock'. Starting countdown!")
            countdown_started = True
            countdown_start_time = time.time()

        # Handle countdown
        if countdown_started:
            elapsed_time = time.time() - countdown_start_time
            countdown = max(0, 5 - int(elapsed_time))

            if countdown > 0:
                cv2.putText(frame, f"Game starts in: {countdown}", (frame_width // 2 - 150, frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # Check for hand movement during the countdown
                for player, current_position in current_positions.items():
                    if has_moved(prev_positions[player], current_position):
                        movement_detected[player] = True
                    prev_positions[player] = current_position

            else:
                print("Countdown complete. Starting the game!")
                # Check if any player didn't move during the countdown
                for player, moved in movement_detected.items():
                    if not moved:
                        print(f"{player.replace('_', ' ').capitalize()} did not move during the countdown! Cheat detected!")
                break

        # Draw annotations for players
        cv2.putText(frame, "Waiting for players to show 'rock'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Waiting for Rock', frame)

        # Exit on spacebar
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break

    cv2.destroyAllWindows()

# Wait for players to show "rock"
wait_for_both_players_to_show_rock()
