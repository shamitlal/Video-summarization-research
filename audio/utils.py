import os

def extract_audio_from_video(video_base_dir,video_name):
    audio_base_dir = "dataset/audio/"
    audio_dir = audio_base_dir + video_name.split(".")[0] + ".wav"
    os.system("ffmpeg -i {0} -f wav -ab 192000 -vn {1}".format(video_base_dir+video_name,audio_dir))


