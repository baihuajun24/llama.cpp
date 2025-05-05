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
        echo "$(date): Push failed. Will retry in 20 seconds."
        sleep 20
    fi
done

echo "Push completed successfully. Exiting script."