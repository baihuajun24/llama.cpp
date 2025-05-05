#!/bin/bash

echo "Starting git push retry script..."
echo "Will attempt to push every 30 minutes until successful."

while true; do
    # Print timestamp
    echo "$(date): Attempting git push..."
    
    # Try to push
    if git push origin main mac-dev; then
        echo "$(date): Push successful!"
        # Exit the loop on success
        break
    else
        echo "$(date): Push failed. Will retry in 30 minutes."
        # Wait for 30 minutes (1800 seconds)
        sleep 1800
    fi
done

echo "Push completed successfully. Exiting script."