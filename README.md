# JETBOT
This is my mods of the Nvidia JetBot 


DISABLE JUPYTR AND CSI SERVICES

sudo systemctl disable jetbot_jupyter.service 

sudo systemctl disable nvargus-daemon.service 

can keep:  jetbot_stats.service


edit  /etc/systemd/system/jetbot_jupyter.service

change to
ExecStart=/usr/bin/python3 /home/jetbot/Notebooks/collision_avoidance/avoid.py



**USE USB CAM**


git clone https://github.com/NVIDIA-AI-IOT/jetcam

cd jetcam

sudo python3 setup.py install

Usage

from jetcam.usb_camera import USBCamera

camera = USBCamera(capture_device=0)



