# JETBOT
This is my mods of the Nvidia JetBot 


**Disable Jupyter and CSI Services**

sudo systemctl disable jetbot_jupyter.service 

sudo systemctl disable nvargus-daemon.service 

can keep:  jetbot_stats.service



**Set the correct startup script**

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


**Add Trims to motors**

/usr/local/lib/python3.6/dist-packages/jetbot-0.3.0-py3.6.egg/jetbot/robot.py


Rtrim = 1.3

Ltrim = 1.0

    def forward(self, speed=1.0, duration=None):
        self.left_motor.value = speed * Ltrim
        self.right_motor.value = speed * Rtrim




