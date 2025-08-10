import pyautogui
import time
from rl_bss import config

def press_key(key):
    """Presses and holds a key."""
    pyautogui.keyDown(key)

def release_key(key):
    """Releases a key."""
    pyautogui.keyUp(key)

def press_and_release(key, duration=config.KEY_PRESS_DURATION):
    """Presses and releases a key for a given duration."""
    press_key(key)
    time.sleep(duration)
    release_key(key)
