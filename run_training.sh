#!/bin/bash
# Evaluation script for obstacle avoidance with both LSTM and Transformer models

# Function to print fancy headers
print_header() {
    echo
    echo "============================================================================="
    echo "$1" | tr '[:lower:]' '[:upper:]' | sed 's/^/ /' | sed 's/$/ /'
    echo "============================================================================="
    echo
}

# Evaluation parameters
EPISODES=500  # Training episodes before evaluation

# Reward configurations
GOALS=(10 20 30)
DISCOMFORTS=(0 -5 -10)
TIMEOUTS=(0 -5 -10)
COLLISIONS=(-10)

# Run evaluation for each reward configuration
print_header "Running Evaluation On Trained Models"
for GOAL in "${GOALS[@]}"; do
    for DISCOMFORT in "${DISCOMFORTS[@]}"; do
        for TIMEOUT in "${TIMEOUTS[@]}"; do
            for COLLISION in "${COLLISIONS[@]}"; do
                REWARD_SET="--reward_goal $GOAL --penalty_discomfort_factor $DISCOMFORT --penalty_timeout $TIMEOUT --penalty_collision $COLLISION"
                REWARD_NAME="goal_${GOAL}_discomfort_${DISCOMFORT}_timeout_${TIMEOUT}_collision_${COLLISION}"

                for MODEL in "lstm" "transformer"; do
                    print_header "Training $(echo $MODEL | tr '[:lower:]' '[:upper:]') model with $REWARD_NAME"
                    
                    python -u train.py --model $MODEL --episodes $EPISODES $REWARD_SET
                done

                print_header "Evaluating with $REWARD_NAME"

                python evaluate.py \
                  --lstm-episode $((EPISODES - 1)) \
                  --transformer-episode $((EPISODES - 1)) \
                  --eval-episodes 100 \
                  --render-every 10 \
                  $REWARD_SET
            done
        done
    done
done

print_header "All Evaluation Complete"
