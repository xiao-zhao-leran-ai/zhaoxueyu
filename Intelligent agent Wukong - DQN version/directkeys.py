# -*- coding: utf-8 -*-
import pyautogui
import time
def D():
    pyautogui.keyDown('d')
    time.sleep(0.5)
    pyautogui.keyUp('d')
def A():
    pyautogui.keyDown('a')
    time.sleep(0.5)
    pyautogui.keyUp('a')
def S():
    pyautogui.keyDown('s')
    time.sleep(0.5)
    pyautogui.keyUp('s')
def W():
    pyautogui.keyDown('w')
    time.sleep(0.5)
    pyautogui.keyUp('w')
def B():
    pyautogui.keyDown('b')
    time.sleep(0.5)
    pyautogui.keyUp('b')
def R():
    pyautogui.keyDown('r')
    time.sleep(0.5)
    pyautogui.keyUp('r')
def left_click(x=None, y=None):
    pyautogui.click(button='left')


def right_click(x=None, y=None):
    pyautogui.click(button='right')


def press_space():
    pyautogui.press('space')

def release_space():
 pyautogui.click(button='left')