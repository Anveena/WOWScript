# coding=utf-8
import datetime
import os
import random
import time
import subprocess
import shlex
import re
import pyautogui as at
from pykeyboard import PyKeyboard
import cv2
from mss import mss
from PIL import Image
import numpy as np
from threading import Thread, Event
import argparse

width_ignore = 1200
bottom_ignore = 540
ext_pixs = 150
thresh = 100
bait_remaining = 0


class AudioHandle(Thread):
    def __init__(self, stop_event, on_fish_event):
        Thread.__init__(self)
        self.event = on_fish_event
        self.stop_event = stop_event

    def run(self):
        command = "ffmpeg -f dshow -i audio=\"立体声混音 (Realtek(R) Audio)\" -ac 2 -ar 192000  -filter_complex ebur128 -f null -"
        p = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        pattern = re.compile(r'M: (.*) S:')
        last_heard_voice = time.time()
        max_voice_this_loop = -100
        total_times = 0
        suc_times = 0
        failed_times = 0
        while p.poll() is None:
            out = p.stdout.readline().strip()
            if time.time() - last_heard_voice > 17:
                failed_times += 1
                total_times += 1
                print(f'\033[91m音频失败率:{failed_times}/{total_times}={(failed_times * 100 / total_times):.1f}%.'
                      f'最大声音:{max_voice_this_loop:.2f}\033[0m')
                last_heard_voice = time.time()
                self.event.set()
                continue
            if out:
                rs = pattern.findall(str(out))
                if len(rs) > 0:
                    current_volume = float(rs[0])
                    if current_volume > max_voice_this_loop:
                        max_voice_this_loop = current_volume
                    if current_volume > -50:
                        if time.time() - last_heard_voice > 2:
                            suc_times += 1
                            total_times += 1
                            last_heard_voice = time.time()
                            print(
                                f'\033[92m音频成功率:{suc_times}/{total_times}={(suc_times * 100 / total_times):.1f}%.'
                                f'当前音量:{current_volume:.2f}\033[0m')
                            self.event.set()
        print(f'\033[91m音频进程已退出\033[0m')
        self.stop_event.set()
        self.event.set()


class VideoHandle(Thread):
    def __init__(self, stop_event, on_fish_event, hit):
        Thread.__init__(self)
        self.event = on_fish_event
        self.stop_event = stop_event
        self.hit = hit
        self.k = PyKeyboard()
        self.template = []
        self.max_loc = []
        self.screenshot_img = []
        self.x_padding = 0
        self.y_padding = 0
        self.suc_times = 0
        self.failed_times = 0
        self.total_times = 0

    def init_float_img(self):
        self.template = cv2.imread('O:\\1.png', 0)

    def process_screenshot(self, tmp):
        tmp_array = np.array(tmp)
        prev_array = np.array(self.screenshot_img)

        tmp_gray = cv2.cvtColor(tmp_array, cv2.COLOR_RGB2GRAY)
        prev_gray = cv2.cvtColor(prev_array, cv2.COLOR_RGB2GRAY)

        _, tmp_binary = cv2.threshold(tmp_gray, thresh, 255, cv2.THRESH_BINARY)
        _, prev_binary = cv2.threshold(prev_gray, thresh, 255, cv2.THRESH_BINARY)

        diff = cv2.absdiff(tmp_binary, prev_binary)
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.screenshot_img = tmp
            self.x_padding = 0
            self.y_padding = 0
            print('\033[91m画面没有变化！\033[0m')
            return
        x_min, y_min = 10000, 10000
        x_max, y_max = 0, 0
        found = False
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 12 or h < 12:
                continue
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            found = True
        if not found:
            self.screenshot_img = tmp
            self.x_padding = 0
            self.y_padding = 0
            print('\033[91m画面基本没有变化！\033[0m')
            return
        self.x_padding = max(0, x_min - ext_pixs)
        self.y_padding = max(0, y_min - ext_pixs)
        print(f'\033[97m未扩展时的处理宽度:{x_max - x_min}高度:{y_max - y_min}\n'
              f'实际处理的水平范围:{self.x_padding + width_ignore}到{min(tmp.width, x_max + ext_pixs) + width_ignore},'
              f'垂直范围:{self.y_padding}到{min(tmp.height, y_max + ext_pixs)}'
              f'\033[0m')
        cropped_region = tmp_array[self.y_padding:min(tmp.height, y_max + ext_pixs),
                         self.x_padding:min(tmp.width, x_max + ext_pixs)]
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
        self.total_times += 1
        if max_val > 0.60:
            self.suc_times += 1
            self.max_loc = (self.max_loc[0] + width_ignore + self.x_padding, self.max_loc[1] + self.y_padding)
            print(
                f'\033[92m视频成功率:{self.suc_times}/{self.total_times}={(self.suc_times * 100 / self.total_times):.1f}%.'
                f'置信度:{max_val:.2f},位置:{self.max_loc[0]},{self.max_loc[1]}\033[0m')
            return True
        self.failed_times += 1
        print(
            f'\033[91m视频失败率:{self.failed_times}/{self.total_times}={(self.failed_times * 100 / self.total_times):.1f}%.'
            f'置信度:{max_val:.2f}\033[0m')
        return False

    def run(self):
        self.init_float_img()
        bait_time = 0
        last_hit_time = 0
        act_start_time = time.time()
        suc_times = 0
        if bait_remaining != 0:
            bait_time = time.time() + 60 * bait_remaining
        while not self.stop_event.is_set():
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
            next_bait_time_interval = time.time() - bait_time
            if next_bait_time_interval > 0:
                self.stop_event.wait(timeout=1)
                print('\033[93m需要使用鱼饵\033[0m')
                self.k.tap_key('e', 1)
                bait_time = time.time() + 600
                print('\033[93m鱼饵已经使用！\033[0m')
                self.stop_event.wait(timeout=3)
            else:
                print(f'\033[93m距离下次使用鱼饵还有{-next_bait_time_interval:.1f}秒\033[0m')
            print('\033[96m截图并释放钓鱼\033[0m')
            self.make_screenshot(True)
            self.stop_event.wait(timeout=0.5)
            self.k.tap_key('1', 1)
            self.event.clear()
            self.stop_event.wait(timeout=2)
            self.make_screenshot(False)
            if self.find_float():
                at.moveTo(self.max_loc[0], self.max_loc[1], duration=0.3)
                print('\033[96m等待鱼上钩的声音\033[0m')
                self.event.wait()
                suc_times += 1
                time_passed_since_act_start = time.time() - act_start_time
                print(f'\033[96m钓到第{suc_times}条鱼,用时{time_passed_since_act_start:.2f}秒.'
                      f'平均每6分钟钓鱼{(360 * suc_times / time_passed_since_act_start):.1f}条\033[0m')
                at.rightClick()
                self.stop_event.wait(timeout=0.2)
                self.k.tap_key('z', 1)
                self.stop_event.wait(timeout=0.4)
                self.k.tap_key('u', 1)
            else:
                self.k.tap_key(' ', 1)
                self.stop_event.wait(timeout=1)
            at.moveTo(random.randint(1, width_ignore), random.randint(1, 1000), duration=0.3)
            print('\033[96m周期结束\n\n周期开始\033[0m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='处理图像参数')
    parser.add_argument('--width_ignore', type=int, default=1200, help='宽度忽略')
    parser.add_argument('--bottom_ignore', type=int, default=540, help='底部忽略')
    parser.add_argument('--ext_pixs', type=int, default=150, help='扩展像素')
    parser.add_argument('--thresh', type=int, default=100, help='二值化的阈值,天亮高,天黑低,降低运算负载度')
    parser.add_argument('--bait_remaining', type=int, default=0, help='当前鱼饵剩余时间(分钟)')
    args = parser.parse_args()
    thresh = args.thresh
    width_ignore = args.width_ignore
    bottom_ignore = args.bottom_ignore
    ext_pixs = args.ext_pixs
    bait_remaining = args.bait_remaining
    stop_event = Event()
    on_fish_event = Event()
    video_thread = VideoHandle(stop_event, on_fish_event, False)
    audio_thread = AudioHandle(stop_event, on_fish_event)
    video_thread.start()
    audio_thread.start()
    try:
        video_thread.join()
        audio_thread.join()
    except KeyboardInterrupt:
        if False:
            print(1)
