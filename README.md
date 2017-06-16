# Senpy Plugins

# Installing


First, install senpy from source or through pip:

    pip install senpy

Each plugin has different requirements.
As of this writing, requirement installation is done manually for each plugin.
All requirements are specified in the .senpy file and, alternatively, in a requirements.txt file.

# Running

Run with:

    git clone https://github.com/MixedEmotions/05_emotion_wassa_nuig
    senpy -f wassaRegression

This will launch the service on http port 5000. You can run `senpy -h` to see other options, including the http port number.
