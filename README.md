# Resistor sorter

Reads the colour bands of through-hole resistors with a Raspberry Pi 4 (a Pi 3
also works, more slowly) and an OBSBOT Meet 2 USB camera, works out the value, and (when running
on the Pi) lights the resistor with NeoPixel strips, turns a servo to send it to the right bin
and shows the bin's number on an 8x8 LED matrix.

```
$ sudo python3 -m resistor_sorter run
preview on http://0.0.0.0:8000/
light: on GPIO 21 (8 LEDs)
servo: on GPIO 18, chute mode
matrix: on /dev/spidev0.0
ready - place a resistor under the camera (Ctrl-C to quit)
4.7 kΩ ±5%  [yellow violet red gold]  -> 1 kΩ – 10 kΩ  (14 of 15 readings)
1 kΩ ±1%  [brown black black brown brown]  -> 1 kΩ – 10 kΩ  (12 of 12 readings)
unreadable (brown (3), grey (19), yellow (20), brown (2))  -> reject
undecided (1 kΩ ±1% x7, 110 Ω ±1% x6)  -> reject
```

It handles 3-, 4-, 5- and 6-band codes, beige and blue bodies, any rotation, and
either end first. Plain OpenCV/NumPy, no machine learning; about 20 ms per frame
on a laptop, so expect very roughly 5–10 readings per second on a Pi 4 (2–4 on a
Pi 3). The live preview runs on its own thread at up to 15 fps whatever the
analysis speed.

## Hardware

| Part | Notes |
| --- | --- |
| Raspberry Pi 4 (or 3) | Raspberry Pi OS Bookworm or newer, 64-bit recommended, Lite is fine |
| OBSBOT Meet 2 USB camera | Focus and zoom are set from the config, see below. Any UVC webcam or a Pi Camera Module also works |
| Plain white matte paper or card | The background the resistor lies on |
| Diffuse light | The NeoPixel strips below, or an LED strip or desk lamp bounced off a white surface |
| Optional: two WS2812 (NeoPixel) strips of 8 LEDs | Lit white by the sorter whenever the camera is in use, so calibration and sorting see the same light. Only driven on a Pi |
| Optional: hobby servo (SG90 etc.) | Aims a chute at the bins, or swings each resistor off to one side or the other. Only driven on a Pi |
| Optional: MAX7219 8x8 LED matrix module | Shows a ring while waiting, a rising fill while a resistor is read, then its bin number (a cross for a reject). SPI; only driven on a Pi |

### Focus and zoom: the one thing that will stop it working

The bands are only readable when the resistor body is at least 100–150 pixels
long and sharp. Autofocus may hunt as resistors come and go, so fix the
focus and zoom by number instead. Both run from 0 to 100 (zoom 0
is the widest view). Try values with the live preview or with snapshots:

```sh
python3 -m resistor_sorter --focus 40 --zoom 30 snap test.jpg
python3 -m resistor_sorter --focus 45 --zoom 30 run    # live, see "Use" below
```

When the bands have crisp edges, put the numbers in `config.toml`:

```toml
[camera]
focus = 45
zoom = 30
```

`--focus -1` gives the autofocus back. The OBSBOT remembers the last focus
and zoom it was given, even after it is unplugged. From Python,
`CvCamera.set_focus(value)` (or `None` for autofocus) and `CvCamera.set_zoom(value)`
change them on an open camera.

### Scene

- Mount the camera looking straight down at the paper, close enough that
  zooming in makes the resistor fill a good part of the frame.
- The paper must fill the whole frame edge to edge: the frame border is used
  to white-balance and to tell what is "background".
- Soft, even light. Hard shadows and bright glare stripes are the main causes
  of misreads. A ring of LEDs or a lamp aimed at the ceiling works well. The
  NeoPixel strips are rows of bright points: bounce them off white card, or put
  a sheet of paper between them and the resistor, rather than aiming them
  straight at it.
- One resistor at a time, fully inside the frame (leads may run off the edge).

## Install on the Pi

```sh
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-spidev v4l-utils
v4l2-ctl --list-devices      # should list "OBSBOT Meet 2"
```

Plug the OBSBOT into one of the blue USB 3 ports. If it is not listed, try
another cable or port. `v4l2-ctl -d /dev/v4l/by-id/usb-Remo_Tech_Co.__Ltd._OBSBOT_Meet_2-video-index0 --list-ctrls`
shows its controls and their current values. (`python3-picamera2` is only
needed for the Pi camera module, `source = "picamera"`.)

For the LED matrix, turn on the SPI port with `sudo raspi-config nonint do_spi 0`
(or `raspi-config` → Interface Options → SPI) and reboot; `ls /dev/spidev*`
should then list `/dev/spidev0.0`. The user created at setup may use it; any
other needs `sudo adduser <user> spi`.

For the NeoPixel light, install Adafruit's library:

```sh
sudo pip3 install --break-system-packages rpi_ws281x adafruit-circuitpython-neopixel
```

It drives the strips from the Pi's PWM and PCM hardware, which only root may
use, so with the light enabled put `sudo` in front of every command below (set
`enabled = false` under `[light]` to do without it). Wire each strip's 5 V and
GND to a 5 V pin (2 or 4) and a ground, and both DINs to GPIO 21 (pin 40); they
show the same colour, so one data line drives both. Sixteen LEDs at full white draw up to about 1 A, which the Pi's 5 V
pins can supply from a 3 A power supply; more strips want a supply of their own
with grounds joined.

The servo runs on the Pi's hardware PWM, which keeps its pulses steady however
busy the CPU is. Turn it on by adding these lines to `/boot/firmware/config.txt`
and rebooting; `ls /sys/class/pwm` should then list `pwmchip0`:

```
dtoverlay=pwm,pin=18,func=2
dtparam=audio=off
```

The second line keeps the onboard audio, which uses the same PWM, away from it.

Copy this folder to the Pi and run everything from inside it with
`python3 -m resistor_sorter ...` (`sudo python3 -m resistor_sorter ...` with the
light). No virtualenv is needed: the apt packages provide everything but the
NeoPixel library. (Installing `opencv-python` from pip on the Pi is slow and can
clash with the apt numpy.)

## Use

**1. Aim and focus.** Start the sorter and open `http://<pi-address>:8000/`
from another computer. The live view shows the detected body (green box), a
dot per band, and the reading. Restart it with different `--focus`/`--zoom`
values until the bands are sharp (see above).

```sh
python3 -m resistor_sorter run
```

**2. Calibrate the colours.** The built-in colour references are rough guesses;
real paint under your light and camera looks different. Calibration is what
makes the readings reliable. Place a resistor you know and name its bands in
either order:

```sh
python3 -m resistor_sorter calibrate yellow violet red gold
python3 -m resistor_sorter calibrate brown black black brown brown
```

Each run stores the measured colour of every band in `calibration.json`, and
those measurements replace the built-in guess for those colours. Do this until
**every colour you expect to meet** has been seen at least once, ideally on
both beige and blue bodies (paint looks different on each). Metal-film 1%
resistors are an easy way to collect brown, black, red and violet.
`calibrate --reset` starts over.

**3. Check.** `read` gives a single reading with details, and `--debug`
saves an annotated picture with the straightened body along the bottom:

```sh
python3 -m resistor_sorter read --debug debug.png          # from the camera
python3 -m resistor_sorter read some_photo.jpg             # from a file
```

```
camera: 4.7 kΩ ±5%  [yellow violet red gold]  margin 34.1, worst band 6 -> trusted
  bands left to right: yellow (6), violet (4), red (4), gold (4)
```

The number after each colour is its distance from the nearest reference
(smaller is better; under ~10 is a good match). A band far from everything
means that colour needs calibrating.

**4. Sort.** `run` watches continuously. No single frame is trusted: a
resistor is watched for 2 seconds and the trustworthy readings from that time
are averaged out by voting. When at least three quarters of them (and at least
3) agree on a bin, and the latest 3 all agree with it as well (so the reading
has settled rather than still changing), the servo turns to that bin's angle,
and the resistor is then ignored until it is taken away. Readings older than 2 seconds are dropped
as it goes, so a bad start while the resistor is being put down does not count
against it. If the readings never agree, it goes to the reject position after
6 seconds and the printed line says which values were read how often. The
light, servo and matrix are only driven on a Raspberry Pi; on a laptop `run`
prints `servo: off (not running on a Raspberry Pi)`, and the like for the other
two, and just reports. An LED matrix, if fitted, shows the bin's number (see
`[matrix]` below).

## Configuration

Copy `config.example.toml` to `config.toml` and edit; it lists every setting
with its default. The main ones:

- **`[[bin]]`**: sort by range (`min`/`max` in ohms) or exact `values`; first
  match wins. Each bin has a servo `angle`.
- **`[servo]`**: `enabled = "auto"` (the default) drives a servo on GPIO 18
  (physical pin 12) when running on a Raspberry Pi with hardware PWM turned on
  (see "Install on the Pi"), and leaves it alone anywhere else; `true` and
  `false` force it. `pin` must be a hardware PWM pin, 12, 13, 18 or 19, and
  match the `pin=` of the `dtoverlay=pwm` line. The default pulse widths of 0.5 and 2.4 ms are an
  SG90's full travel; narrow them towards 1 and 2 ms for a servo that strains
  at ±90°.
  `mode = "chute"` turns to the bin's angle and stays there until the next
  resistor. `mode = "swing"` turns there, holds that angle for `hold` seconds
  (3 by default) and turns back to `rest_angle`, for a paddle that tips or
  pushes each resistor off to one side or the other. Power the servo from its
  own 5 V supply with grounds joined; a servo on the Pi's 5 V pin can brown it
  out.
- **`[matrix]`**: an 8x8 LED matrix on a MAX7219 shows a ring while the sorter
  waits, fills up from the bottom while a resistor is watched, and then shows
  its bin number (the bin's position in this file, 1 for the first `[[bin]]`;
  two small digits from 10 up), or a cross for a reject, until the resistor is
  taken away. `enabled` works as for the servo: `"auto"` drives it on a Pi with
  `python3-spidev` installed. Wire VCC to 5 V (pin 2), GND to ground (pin 6),
  DIN to GPIO 10/MOSI (pin 19), CS to GPIO 8/CE0 (pin 24) and CLK to
  GPIO 11/SCLK (pin 23). The module wants 5 V; the Pi's 3.3 V signals are a
  little under what the chip's datasheet asks for but work in practice, and
  nothing is sent back to the Pi, so no level shifter is needed. `brightness`
  goes from 0 to 15. The module can be mounted any way round: run
  `python3 -m resistor_sorter matrix`, which shows the ready ring, each bin's
  number and the reject cross in turn, and set `rotate` (0, 90, 180 or 270
  degrees clockwise) and `mirror` until they read correctly. On a "4-in-1"
  module set `chain = 4`; only the block nearest the input connector is used.
- **`[light]`**: WS2812 (NeoPixel) strips that light the resistor. They come on
  white whenever the camera is in use (`run`, `read`, `snap` and `calibrate`)
  and go off at exit, so calibration and sorting see the same light. `enabled`
  works as for the servo. `pins` lists the GPIO of each data line (21; GPIO 12
  and 18 would take the servo's PWM and 10 the matrix's SPI), `leds` is how many LEDs
  each strip has, `brightness` goes from 0 to 1 and `color` is the red, green
  and blue to show. The colour is resent once a second, so a strip that lost
  it to a dip in its supply recovers. Installing the library and wiring are
  under "Install on the Pi".
- **`[sorter]`**: `window` is how many seconds of readings each decision is
  based on, `min_readings` how many of them it needs, and `min_agreement` the
  share that must point at the same bin. `stable_readings` is how many of the
  latest readings must all point at that bin as well. `reject_after` and `clear_after` are
  the seconds before giving up on a resistor and before the next one is
  expected. `min_margin` and `max_color_distance` decide when a single reading
  counts as trustworthy. Raising `min_agreement`, `min_readings`,
  `stable_readings` or `min_margin`, or lowering `max_color_distance`, gives
  fewer wrong bins and more rejects.
- **`[web]`**: the preview has no password. Set `host = "127.0.0.1"` or
  `enabled = false` on a shared network.
- **`[camera]`**: `source` is the OBSBOT's `/dev/v4l/by-id/` path by
  default. Use `"0"` for another webcam, `"picamera"` for a Pi camera module
  (with `width = 1296`, `height = 972`), or an image/video file to try things on
  a laptop. `focus` and `zoom` are described above. `lock_after` is how
  long exposure and white balance get to settle before frames are used. A Pi
  camera then freezes them. USB cameras such as the OBSBOT keep them automatic,
  because they do not report the values auto mode settled on, and the paper-based
  white balance below cancels the drift; the camera is also put back to
  automatic (and its old zoom) when the program exits.
  The default 3840x2160 (4K, 30 fps) MJPEG is costly to decode on a Pi; if
  readings come too slowly, use 1920x1080 or 1280x720 and zoom in further.

For the simplest rig, a servo that swings low values one way and high values
the other, use two bins and `swing` mode:

```toml
[servo]
mode = "swing"     # turn, hold for 3 seconds, turn back to the middle

[[bin]]
name = "under 1 kΩ"
max = 1000
angle = -60

[[bin]]
name = "1 kΩ and up"
min = 1000
angle = 60
```

To start the sorter at boot, create `/etc/systemd/system/resistor-sorter.service`:

```ini
[Unit]
Description=Resistor sorter
After=network.target

[Service]
WorkingDirectory=/home/pi/resistor-sorter
ExecStart=/usr/bin/python3 -m resistor_sorter run
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

then `sudo systemctl enable --now resistor-sorter`, and follow the log with
`journalctl -fu resistor-sorter`. The service runs as root, which the NeoPixel
light needs; with the light disabled, add `User=pi` under `[Service]`.

## How it works

1. **Even out the light.** The paper's colour across the frame is fitted as a
   smooth surface, with the resistor and any clutter rejected as outliers, and
   divided out. The paper becomes even neutral grey, which cancels the colour
   of the light, darker corners and a lamp off to one side.
2. **Find the body.** Pixels that differ from the paper are foreground. The
   leads are much thinner than the body, so a morphological opening sized to
   the body's thickness removes them. Pale bands (white, silver) that cut the
   body in pieces are bridged first by a closing.
3. **Straighten.** A rotated rectangle around the body is turned (by up to
   40°) to the angle at which the bands look sharpest, since a shadow or stain
   touching the body skews it, then warped into a horizontal strip. Inside
   the strip, the body's rows are the ones the band pattern runs across, and
   its ends are where it fills those rows.
4. **Split into bands.** The middle half of the body's rows gives a colour
   profile along it. The body colour is the one at both ends (the end caps
   carry no paint) and covering the most length; if a second colour comes
   close, both are tried and the decoder picks the one that reads better. The
   bands are the stretches that stand out from the body colour. Being lighter
   than the body counts for less, because glare brightens bare body. A dip in
   contrast splits two bands whose blurred gap never quite returns to body
   colour. Each band's colour is compared, in CIELAB space, with the
   calibrated references, allowing for some of the body's colour bleeding into
   a narrow band.
5. **Decode.** Every combination of each band's top three colour candidates is
   tried in both reading directions. Invalid codes are thrown out (gold cannot
   be a digit, the first digit cannot be black...). The rest are scored by
   colour distance, plus penalties for:
   - values outside the standard E24/E96 series;
   - band spacing that says the resistor is backwards (the tolerance band sits
     apart from the rest), in proportion to how lopsided the spacing is;
   - three-band codes, which are rare and what a code with missed bands leaves;
   - skipping a stretch at the very end, which may be end cap rather than
     paint, or a faint one in between, which may be glare.

   The best score wins. `margin` is how far ahead it is of the best reading
   with a different value.

## Limitations

- Reliable colour naming depends on calibration under your own light. With
  only the built-in references, expect many readings to be refused. That is
  deliberate: refusing is better than sorting into the wrong bin.
- A band the same colour as the body (a blue band on a blue body, sometimes
  gold on a tan body) is invisible to step 4.
- Many 5-band codes read validly in both directions: 1 kΩ ±1% backwards is
  110 Ω ±1%, 10 kΩ ±1% is 120 Ω ±1%, 220 Ω ±1% is 10 kΩ ±2%. Only the spacing
  tells them apart, and on some cheap metal film resistors the bands are
  almost evenly spaced. Such resistors are rejected rather than guessed.
- Motion blur and soft focus merge narrow gaps, so two adjacent bands of the
  same colour (red red ...) can be read as one. Hold the camera still and fix
  the focus.
- The Pi camera code (`camera.PiCamera`), servo code (`sorter.Chute`) and LED
  matrix code (`matrix.Max7219`) follow the picamera2 and Linux PWM
  documentation and the MAX7219 datasheet but were written without the
  hardware at hand. The light code (`light.Light`) makes the same calls as a
  short script run on the rig's two strips, but has not been driven from the
  sorter itself yet. The USB camera code (4K capture,
  focus, zoom and the exposure/white balance lock) was checked on an OBSBOT
  Meet 2. The vision and sorting logic is covered by tests.

## Development

On a laptop, with [uv](https://docs.astral.sh/uv/):

```sh
uv run pytest                                  # tests (synthetic images)
uv run resistor-sorter --source photo.jpg run  # the full loop on a still image (no servo off the Pi)
```

To check against the library versions Raspberry Pi OS Bookworm ships:

```sh
uv run --isolated --python 3.11 --with numpy==1.24.2 \
  --with opencv-python-headless==4.6.0.66 --with pytest --with-editable . pytest
```

| File | What it does |
| --- | --- |
| `resistor_sorter/vision.py` | finding the body and measuring bands |
| `resistor_sorter/decode.py` | colour codes to ohms, choosing the likeliest reading |
| `resistor_sorter/colors.py` | code tables, colour references, calibration file |
| `resistor_sorter/sorter.py` | when to accept/reject, bin choice, servo |
| `resistor_sorter/matrix.py` | MAX7219 LED matrix: the signs it shows, and the SPI driver |
| `resistor_sorter/light.py` | NeoPixel strips that light the resistor while the camera is in use |
| `resistor_sorter/camera.py` | USB camera (focus, zoom, exposure lock), Pi camera and file sources |
| `resistor_sorter/web.py` | MJPEG live preview |
| `resistor_sorter/cli.py` | `run`, `read`, `calibrate`, `snap`, `matrix` |
| `tests/synth.py` | draws synthetic resistor photos for the tests, optionally with blur, uneven light, shadows and JPEG |
