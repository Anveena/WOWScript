# coding=utf-8
import random
import time
import subprocess, shlex
from threading import Thread, Event

import re
import autopy as at
from pykeyboard import PyKeyboard
import cv2
from mss import mss
from PIL import Image
import numpy as np


class AudioHandle(Thread):
    def __init__(self, on_fish_event):
        Thread.__init__(self)
        self.event = on_fish_event

    def run(self):
        print('starting audio handle')
        command = "ffmpeg -f avfoundation -i :0 -af astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level  -f null aa"
        p = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        pattern = re.compile(r'RMS_level=(.*)\'')
        last_heard_voice = 0
        while p.poll() is None:
            out = p.stdout.readline().strip()
            if time.time() - last_heard_voice > 26:
                print('好久没听到钓鱼声了.先放过')
                last_heard_voice = time.time()
                self.event.set()
                continue
            if out:
                rs = pattern.findall(str(out))
                if len(rs) > 0:
                    current_volume = float(rs[0])
                    if current_volume > -10:
                        if time.time() - last_heard_voice > 2:
                            last_heard_voice = time.time()
                            print('钓到鱼了,音量:', current_volume)
                            self.event.set()


class VideoHandle(Thread):
    def __init__(self, on_fish_event, hit):
        Thread.__init__(self)
        self.event = on_fish_event
        self.hit = hit
        self.k = PyKeyboard()
        self.template = []
        self.max_loc = []
        self.screenshot_img = []

    def init_float_img(self):
        self.template = cv2.imread('/Users/panys/PycharmProjects/WOWScript/float.png', 0)

    def make_screenshot(self):
        with mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            self.screenshot_img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        return

    def find_float(self):
        img_rgb = np.array(self.screenshot_img)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, self.max_loc = cv2.minMaxLoc(res)
        if max_val > 0.6:
            print('找到了,置信度是', max_val)
            return True
        print('没有找到', max_val)
        return False

    def run(self):
        self.init_float_img()
        start_time = time.time()
        last_hit_time = time.time()
        while True:
            if self.hit and time.time() - last_hit_time > 600:
                print('需要召唤生物击杀')
                self.k.tap_key('e', 1)
                time.sleep(5)
                self.k.tap_key('q', 1)
                time.sleep(3)
                self.k.tap_key('g', 1)
                time.sleep(5)
                last_hit_time = time.time()
            if time.time() - start_time > 500:
                print('需要使用鱼饵')
                self.k.tap_key('5', 1)
                start_time = time.time()
                print('鱼饵已经使用!')
                time.sleep(2)
            time.sleep(1)
            print('准备钓鱼')
            self.k.tap_key('q', 1)
            self.event.clear()
            print('吊钩已放')
            time.sleep(2.5)
            print('准备截图')
            self.make_screenshot()
            print('截图完成')
            if self.find_float():
                print(self.max_loc)
                at.mouse.smooth_move(self.max_loc[0] / 2, self.max_loc[1] / 2)  # 这里说明找到了
                print('等待钓鱼')
                self.event.wait()
                at.mouse.click(at.mouse.Button.RIGHT)
            else:
                self.k.tap_key(' ', 1)
            at.mouse.smooth_move(random.randint(1, 1000), random.randint(1, 1439))


if __name__ == '__main__':
    on_fish_event = Event()
    video_thread = VideoHandle(on_fish_event,True)
    audio_thread = AudioHandle(on_fish_event)
    video_thread.start()
    audio_thread.start()
    video_thread.join()
    audio_thread.join()
    at.mouse.smooth_move(random.randint(1, 1000), random.randint(1, 1439))
