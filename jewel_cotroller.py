import board
import neopixel

# === Lux-Calibrated Color Presets ===
COLOR_PRESETS = {
    "blue_7.5": {"color": (0, 0, 255), "brightness": 0.05},
    "blue_30": {"color": (0, 0, 255), "brightness": 0.20},
    "red_7.5": {"color": (255, 0, 0), "brightness": 0.01},
    "red_30": {"color": (255, 0, 0), "brightness": 0.04},
    "green_7.5": {"color": (0, 255, 0), "brightness": 0.0023},
    "green_30": {"color": (0, 255, 0), "brightness": 0.0094},
}

# === Jewel Control Function ===
def apply_color(jewel_pin, preset_key):
    config = COLOR_PRESETS[preset_key]
    color = config["color"]
    brightness = config["brightness"]

    strip = neopixel.NeoPixel(jewel_pin, 7, brightness=brightness, auto_write=False)
    for i in range(7):
        strip[i] = color
    strip.show()

# === Example Usage ===
# Apply blue at 7.5 lux to the first Jewel on pin A3
apply_color(board.A3, "blue_7.5")

# Apply green at 30 lux to the second Jewel on pin A1
apply_color(board.A1, "green_30")
