import cv2
import time
from ultralytics import YOLO

from ultralytics.utils import LOGGER

LOGGER.setLevel("ERROR")
model = YOLO('.\\RockPaperScissors\\Weights\\best.pt', verbose=False)

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

player_1_score = 0
player_2_score = 0
first_round = True


def has_moved_during_countdown(initial_position, position_history, threshold):
    if initial_position is None or not position_history:
        return False
    x1, y1 = initial_position
    for (x2, y2) in position_history:
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distance > threshold:
            return True
    return False


def determine_winner(player_1_choice, player_2_choice):
    global player_1_score, player_2_score
    outcomes = {
        ('rock', 'scissor'): 'Player 1',
        ('scissor', 'paper'): 'Player 1',
        ('paper', 'rock'): 'Player 1',
        ('scissor', 'rock'): 'Player 2',
        ('paper', 'scissor'): 'Player 2',
        ('rock', 'paper'): 'Player 2',
    }
    if player_1_choice == player_2_choice:
        print("It's a tie!")
    else:
        winner = outcomes.get((player_1_choice, player_2_choice))
        if winner == 'Player 1':
            player_1_score += 1
            print("Player 1 wins this round!")
        elif winner == 'Player 2':
            player_2_score += 1
            print("Player 2 wins this round!")


def reduce_score(player_key):
    global player_1_score, player_2_score
    if player_key == "player_1":
        player_1_score = max(0, player_1_score - 1)  # Ensure score doesn't go below 0
    elif player_key == "player_2":
        player_2_score = max(0, player_2_score - 1)  # Ensure score doesn't go below 0
    print(f"{player_key.replace('_', ' ').title()} cheated! -1 point.")


def play_round(first_round):
    print(f"{'Waiting for both players to show rock...' if first_round else 'Starting next round!'}")

    countdown_started = False
    countdown_start_time = None
    countdown_complete_time = None
    hand_state_locked_time = None

    prev_states = {"player_1": None, "player_2": None}
    current_states = {"player_1": None, "player_2": None}
    prev_positions = {"player_1": None, "player_2": None}
    current_positions = {"player_1": None, "player_2": None}
    position_history = {"player_1": [], "player_2": []}

    player_1_choice = None
    player_2_choice = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 0, 0), 2)

        results = model(frame)
        detections = results[0]

        if detections.boxes is not None:
            for box in detections.boxes:
                if box.conf[0] < 0.7:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                class_name = model.names[cls].lower()

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                player_key = "player_1" if center_x < frame_width // 2 else "player_2"

                current_states[player_key] = class_name
                current_positions[player_key] = (center_x, center_y)

                if player_key == "player_1":
                    player_1_choice = class_name
                else:
                    player_2_choice = class_name

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{player_key.replace('_', ' ').title()}: {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if first_round and player_1_choice == "rock" and player_2_choice == "rock" and not countdown_started:
            print("Both players showed 'rock'. Starting countdown!")
            countdown_started = True
            countdown_start_time = time.time()
            prev_positions = current_positions.copy()
        elif not first_round and not countdown_started:
            print("Skipping 'rock' requirement. Starting countdown!")
            countdown_started = True
            countdown_start_time = time.time()
            prev_positions = current_positions.copy()

        if countdown_started:
            elapsed_time = time.time() - countdown_start_time
            countdown = max(0, 5 - int(elapsed_time))

            if countdown > 0:
                for player in ["player_1", "player_2"]:
                    if current_positions[player]:
                        position_history[player].append(current_positions[player])

                cv2.putText(frame, f"Game starts in: {countdown}", (frame_width // 2 - 150, frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                if countdown_complete_time is None:
                    countdown_complete_time = time.time()
                    hand_state_locked_time = time.time()
                    prev_states = current_states.copy()

                    cheat_messages = []
                    for player in ["player_1", "player_2"]:
                        if first_round:
                            if not has_moved_during_countdown(prev_positions[player], position_history[player],
                                                              threshold=50):
                                cheat_messages.append(f"{player.replace('_', ' ').title()} didn't move!")
                                reduce_score(player)

                    if cheat_messages:
                        cheat_message = " and ".join(cheat_messages)
                        print(f"Cheating detected: {cheat_message}")

                        cv2.putText(frame, cheat_message, (frame_width // 2 - 200, frame_height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        return

                if time.time() - hand_state_locked_time <= 4:
                    cv2.putText(frame, "Don't change your hand state!", (frame_width // 2 - 150, frame_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    for player in ["player_1", "player_2"]:
                        if prev_states[player] and prev_states[player] != current_states[player]:
                            print(f"{player.replace('_', ' ').title()} changed hand state! Cheating detected.")
                            reduce_score(player)  # Deduct 1 point for the player who cheated
                            return
                else:
                    determine_winner(player_1_choice, player_2_choice)
                    return

        cv2.imshow('Rock-Paper-Scissors', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


player_1_score = 0
player_2_score = 0
required_rounds = int(input("Enter number of rounds: "))
while True:

    first_round = True

    for i in range(required_rounds):
        print("round: " + str(i+1))
        play_round(first_round)
        first_round = False

        if player_1_score == required_rounds:
            print(f" Player 1 wins the game with {player_1_score} points!")
            exit()
        elif player_2_score == required_rounds:
            print(f" Player 2 wins the game with {player_2_score} points!")
            exit()

        print("Press space to go to the next round!")
        while True:
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        print("Starting next round...")

    print("No winner, restarting game...")