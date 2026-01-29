source saffi/bin/activate

cd project_v2

sudo systemctl restart ollama  

export OLLAMA_NUM_THREADS=4

export OLLAMA_MAX_LOADED_MODELS=1



python main.py


v4l2-ctl --list-devices  #to check the video port
ffplay /dev/videox        #to play the video
 



ollama serve 