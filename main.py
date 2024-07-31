# coding=utf-8
import datetime
import random
import time
import subprocess, shlex
from threading import Thread, Event

import re
import pyautogui as at
from pykeyboard import PyKeyboard
import cv2
from mss import mss
from PIL import Image
import numpy as np

width_ignore = 1200
bottom_ignore = 540
ext_pixs = 100


class AudioHandle(Thread):
    def __init__(self, on_fish_event):
        Thread.__init__(self)
        self.event = on_fish_event

    def run(self):
        print('starting audio handle')
        command = "ffmpeg -f dshow -i audio=\"立体声混音 (Realtek(R) Audio)\" -ac 2 -ar 192000  -filter_complex ebur128 -f null -"
        p = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        pattern = re.compile(r'M: (.*) S:')
        last_heard_voice = 0
        max_voice_this_loop = -100
        while p.poll() is None:
            out = p.stdout.readline().strip()
            if time.time() - last_heard_voice > 17:
                print('好久没听到钓鱼声了.最大声音', max_voice_this_loop)
                last_heard_voice = time.time()
                self.event.set()
                continue
            if out:
                rs = pattern.findall(str(out))
                if len(rs) > 0:
                    current_volume = float(rs[0])
                    if current_volume > max_voice_this_loop:
                        max_voice_this_loop = current_volume
                    if current_volume > -50:  # 56.31,51.91
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
        self.x_padding = 0
        self.y_padding = 0

    def init_float_img(self):
        self.template = cv2.imread('O:\\1.png', 0)

    #  process_screenshot chatGPT生成的,提示词:
    #  我是用的是 import cv2
    # 1.二值化tmp和self.screenshot_img.生成两个新图片
    # 2.比较并选出两个新图片中不同的区域 给出x,y,width,height
    # 3.裁剪tmp为不同的区域.
    # 4.保存tmp为jpeg,在O盘下,以时间戳命名
    # 5.self.screenshot_img = tmp
    def process_screenshot(self, tmp):
        # 1. Convert PIL Image to numpy array
        tmp_array = np.array(tmp)
        prev_array = np.array(self.screenshot_img)

        # 2. Convert to grayscale
        tmp_gray = cv2.cvtColor(tmp_array, cv2.COLOR_RGB2GRAY)
        prev_gray = cv2.cvtColor(prev_array, cv2.COLOR_RGB2GRAY)

        # 3. Threshold to binary images
        _, tmp_binary = cv2.threshold(tmp_gray, 128, 255, cv2.THRESH_BINARY)
        _, prev_binary = cv2.threshold(prev_gray, 128, 255, cv2.THRESH_BINARY)

        # 4. Find differences
        diff = cv2.absdiff(tmp_binary, prev_binary)
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store the bounding rectangle coordinates
        if contours:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 20 or h < 20:
                    continue
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            # 5. Crop the different regions
            self.x_padding = max(0, x_min - ext_pixs)
            self.y_padding = max(0, y_min - ext_pixs)
            cropped_region = tmp_array[self.y_padding:min(tmp.height, y_max + ext_pixs),
                             self.x_padding:min(tmp.width, x_max + ext_pixs)]

            # 6. Save the cropped region as JPEG with a timestamp
            # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            # save_path = os.path.join("O:\\", f"{timestamp}.jpg")
            # cv2.imwrite(save_path, cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR))
            # diff_save_path = os.path.join("O:\\", f"{timestamp}_diff.jpg")
            # cv2.imwrite(diff_save_path, diff)
            self.screenshot_img = Image.fromarray(cropped_region)

    def make_screenshot(self, is_first):
        with mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            tmp = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
            tmp = tmp.crop((width_ignore, 0, tmp.width - width_ignore, tmp.height - bottom_ignore))
            if is_first:
                self.screenshot_img = tmp
            else:
                self.process_screenshot(tmp)
        return

    def find_float(self):
        img_rgb = np.array(self.screenshot_img)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, self.max_loc = cv2.minMaxLoc(res)
        if max_val > 0.60:
            print('找到了,置信度是', max_val, '位置在:', self.max_loc[0], self.max_loc[1])
            return True
        print('没有找到', max_val)
        return False

    def run(self):
        self.init_float_img()
        start_time = 0
        last_hit_time = 0
        while True:
            # if self.hit and time.time() - last_hit_time > 444:
            #     print('需要召唤生物击杀')
            #     self.k.tap_key('e', 1)
            #     time.sleep(5)
            #     self.k.tap_key('q', 1)
            #     time.sleep(3)
            #     self.k.tap_key('g', 1)
            #     time.sleep(5)
            #     last_hit_time = time.time()
            # else:
            #     print('还在时间内,不需要打怪', time.time() - last_hit_time)
            target_time = datetime.datetime(2024, 7, 24, 17, 20)
            if time.time() - start_time > 600 and datetime.datetime.fromtimestamp(time.time()) > target_time:
                time.sleep(3)
                print('需要使用鱼饵')
                self.k.tap_key('e', 1)
                start_time = time.time()
                print('鱼饵已经使用!')
                time.sleep(3)
            else:
                print('还在时间内,不需要使用鱼饵', time.time() - start_time)
            print('准备钓鱼')
            self.make_screenshot(True)
            time.sleep(0.5)
            self.k.tap_key('1', 1)
            self.event.clear()
            time.sleep(2)
            self.make_screenshot(False)
            if self.find_float():
                print(self.x_padding, self.y_padding)
                at.moveTo(self.max_loc[0] + width_ignore + self.x_padding, self.max_loc[1] + self.y_padding,
                          duration=0.3)  # 这里说明找到了
                print('等待钓鱼')
                self.event.wait()
                at.rightClick()
                time.sleep(0.2)
                self.k.tap_key('z', 1)
                time.sleep(0.4)
                self.k.tap_key('u', 1)
            else:
                self.k.tap_key(' ', 1)
                time.sleep(1)
            at.moveTo(random.randint(1, width_ignore), random.randint(1, 1000), duration=0.3)


if __name__ == '__main__':
    on_fish_event = Event()
    video_thread = VideoHandle(on_fish_event, False)
    audio_thread = AudioHandle(on_fish_event)
    video_thread.start()
    audio_thread.start()
    video_thread.join()
    audio_thread.join()
