# Resistor sorter

Reads the colour bands of through-hole resistors with a Raspberry Pi 4 (a Pi 3
also works, more slowly) and an OBSBOT Meet 2 USB camera, works out the value, and (optionally) swings a
servo-driven chute to drop each resistor into the right bin.

```
$ python3 -m resistor_sorter run
preview on http://0.0.0.0:8000/
ready - place a resistor under the camera (Ctrl-C to quit)
4.7 kΩ ±5%  [yellow violet red gold]  -> 1 kΩ – 10 kΩ
1 kΩ ±1%  [brown black black brown brown]  -> 1 kΩ – 10 kΩ
unreadable (brown (3), grey (19), yellow (20), brown (2))  -> reject
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
| Diffuse light | LED strip or desk lamp bounced off a white surface |
| Optional: hobby servo (SG90 etc.) | Aims a chute at the bins |

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
  of misreads. A ring of LEDs or a lamp aimed at the ceiling works well.
- One resistor at a time, fully inside the frame (leads may run off the edge).

## Install on the Pi

```sh
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-gpiozero v4l-utils
v4l2-ctl --list-devices      # should list "OBSBOT Meet 2"
```

Plug the OBSBOT into one of the blue USB 3 ports. If it is not listed, try
another cable or port. `v4l2-ctl -d /dev/v4l/by-id/usb-Remo_Tech_Co.__Ltd._OBSBOT_Meet_2-video-index0 --list-ctrls`
shows its controls and their current values. (`python3-picamera2` is only
needed for the Pi camera module, `source = "picamera"`.)

Copy this folder to the Pi and run everything from inside it with
`python3 -m resistor_sorter ...`. No virtualenv or pip install is needed: the
apt packages provide everything. (Installing `opencv-python` from pip on the Pi
is slow and can clash with the apt numpy.)

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

**4. Sort.** `run` watches continuously. A resistor is accepted once the same
value is read on 5 frames in a row, sent to the first bin that takes it, and
then ignored until it is removed. If no trustworthy reading comes within 30
frames it goes to the reject position.

## Configuration

Copy `config.example.toml` to `config.toml` and edit; it lists every setting
with its default. The main ones:

- **`[[bin]]`**: sort by range (`min`/`max` in ohms) or exact `values`; first
  match wins. Each bin has a servo `angle`.
- **`[servo]`**: set `enabled = true` to drive a chute servo from GPIO 18
  (physical pin 12). Power the servo from its own 5 V supply with grounds
  joined; a servo on the Pi's 5 V pin can brown it out. The servo is detached
  after each move so it does not buzz.
- **`[sorter]`**: `min_margin` and `max_color_distance` decide when a reading
  counts as trustworthy. Raising `min_margin` or lowering `max_color_distance`
  gives fewer wrong bins and more rejects.
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

To start the sorter at boot, create `/etc/systemd/system/resistor-sorter.service`:

```ini
[Unit]
Description=Resistor sorter
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/resistor-sorter
ExecStart=/usr/bin/python3 -m resistor_sorter run
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

then `sudo systemctl enable --now resistor-sorter`, and follow the log with
`journalctl -fu resistor-sorter`.

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
- The Pi camera code (`camera.PiCamera`) and servo code (`sorter.Chute`)
  follow the picamera2 and gpiozero documentation but were written without the
  hardware at hand. The USB camera code (4K capture, focus, zoom and the
  exposure/white balance lock) was checked on an OBSBOT Meet 2. The vision and sorting logic is covered by tests.

## Development

On a laptop, with [uv](https://docs.astral.sh/uv/):

```sh
uv run pytest                                  # tests (synthetic images)
uv run resistor-sorter --source photo.jpg run  # the full loop on a still image
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
| `resistor_sorter/camera.py` | USB camera (focus, zoom, exposure lock), Pi camera and file sources |
| `resistor_sorter/web.py` | MJPEG live preview |
| `resistor_sorter/cli.py` | `run`, `read`, `calibrate`, `snap` |
| `tests/synth.py` | draws synthetic resistor photos for the tests, optionally with blur, uneven light, shadows and JPEG |
