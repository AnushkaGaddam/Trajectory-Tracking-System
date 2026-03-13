
from vpython import *
import math, random, time, threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np

# ══════════════════════════════════════════════════════════════
#  WORLD SCALE
# ══════════════════════════════════════════════════════════════
UNIT_TO_M     = 1.0
SEA_SURFACE_Y = 0.0
SEA_FLOOR_Y   = -30.0
PIPE_Y        = 60.0

HORIZ_PIPE_X0 = -75.0
HORIZ_PIPE_X1 =  75.0
HORIZ_ZIG     = 10.0

VERT_PIPE_X   = HORIZ_PIPE_X1
VERT_PIPE_Z   = 0.0
VERT_PIPE_BOT = PIPE_Y
VERT_PIPE_TOP = PIPE_Y + 100.0

PIPE_RADIUS   = 1.2

BASE_STEP     = 0.32
DT            = 0.020
OBS_BUFFER    = 7.0
ALERT_DIST    = 13.0
SIDESTEP_W    = 5.0
BATTERY_DRAIN = 0.002

HELIX_RADIUS      = 5.0
HELIX_TURNS       = 2.5
HELIX_STEPS       = 100
VERT_HELIX_RADIUS = 4.2
VERT_HELIX_TURNS  = 3.0
VERT_HELIX_STEPS  = 120

speed_mult = 1.0

PIPE_SEG_TEMPS = [118.5, 74.2, 91.8, 55.3, 108.7, 63.1]
VERT_SEG_TEMPS = [ 48.6, 57.4, 72.1, 85.3,  61.9]

PIPE_TEMP_ALERT_HOT  = 105.0
PIPE_TEMP_WARN_HOT   =  85.0
PIPE_TEMP_WARN_COOL  =  50.0
PIPE_TEMP_ALERT_COOL =  40.0

def pipe_temp_to_color(t_c):
    if   t_c >= 110: return vector(1.00, 0.05, 0.00)
    elif t_c >=  90: return vector(1.00, 0.35, 0.00)
    elif t_c >=  70: return vector(1.00, 0.72, 0.00)
    elif t_c >=  55: return vector(0.95, 0.95, 0.15)
    elif t_c >=  45: return vector(0.60, 0.90, 0.60)
    elif t_c >=  35: return vector(0.10, 0.82, 1.00)
    else:            return vector(0.15, 0.40, 1.00)

def pipe_temp_label(t_c):
    if t_c >= PIPE_TEMP_ALERT_HOT:  return "CRITICAL HOT "
    if t_c >= PIPE_TEMP_WARN_HOT:   return "WARNING  HOT "
    if t_c >= PIPE_TEMP_WARN_COOL:  return "NORMAL       "
    if t_c >= PIPE_TEMP_ALERT_COOL: return "WARNING  COOL"
    return                                  "CRITICAL COOL"

def pipe_temp_hud_color(t_c):
    if t_c >= PIPE_TEMP_ALERT_HOT:  return vector(1.00, 0.10, 0.00)
    if t_c >= PIPE_TEMP_WARN_HOT:   return vector(1.00, 0.60, 0.10)
    if t_c >= PIPE_TEMP_WARN_COOL:  return vector(0.20, 1.00, 0.55)
    if t_c >= PIPE_TEMP_ALERT_COOL: return vector(0.40, 0.80, 1.00)
    return                                  vector(0.15, 0.40, 1.00)

CURRENT_ZONES = [
    (-45, 0, 22, 0.18, vector( 1.0, 0,  0.4)),
    (  0, 0, 18, 0.22, vector(-0.6, 0,  1.0)),
    ( 45, 0, 20, 0.20, vector( 0.5, 0, -1.0)),
]
VERT_CURRENT_ZONE = (VERT_PIPE_X, VERT_PIPE_Z, 12, 0.25, vector(0.4, 0.6, 0.3))

BASE_TURBIDITY = 18.0
turbidity      = BASE_TURBIDITY
cam_mode       = 1
paused         = False
replay_requested  = False
reset_requested   = False
current_event_log = []
total_current_hits= 0

# ══════════════════════════════════════════════════════════════
#  SCENE
# ══════════════════════════════════════════════════════════════
scene = canvas(
    title=(
        "<div style='"
        "background:linear-gradient(90deg,#003855 0%,#005f7a 50%,#003855 100%);"
        "padding:8px 20px;"
        "border-bottom:2px solid #00c8e0;"
        "font-family:monospace;font-size:14px;"
        "color:#b0f0ff;letter-spacing:3px;font-weight:bold;'>"
        "&nbsp; TRAJECTORY TRACKING SYSTEM &nbsp;|&nbsp; 3D &nbsp;|"
        "</div>"
    ),
    width=1280, height=680,
    background=vector(0.012, 0.18, 0.32),
    userzoom=True, userspin=True,
)
scene.camera.pos = vector(0, 80, 120)
scene.forward    = vector(0, -0.35, -1)
scene.up         = vector(0, 1, 0)

distant_light(direction=vector( 0.2,  1.0,  0.4), color=vector(0.25, 0.55, 0.72))
distant_light(direction=vector(-0.6,  0.3,  0.5), color=vector(0.05, 0.18, 0.32))
distant_light(direction=vector( 0.1, -0.6,  0.3), color=vector(0.02, 0.09, 0.18))

# ── UI buttons ────────────────────────────────────────────────
def toggle_pause_btn(b):
    global paused
    paused = not paused
    if paused:
        b.text='  ▶  RESUME  '; b.background=color.orange; b.color=color.black
    else:
        b.text='  ⏸  PAUSE  '
        b.background=vector(0.0,0.18,0.30); b.color=vector(0.0,0.90,1.00)

def do_replay(b): global replay_requested; replay_requested = True
def do_reset(b):  global reset_requested;  reset_requested  = True

def handle_key(evt):
    global paused, cam_mode
    k = evt.key
    if k == ' ':
        paused = not paused
        try:
            if paused:
                pause_btn.text='  ▶  RESUME  '; pause_btn.background=color.orange; pause_btn.color=color.black
            else:
                pause_btn.text='  ⏸  PAUSE  '
                pause_btn.background=vector(0.0,0.18,0.30); pause_btn.color=vector(0.0,0.90,1.00)
        except: pass
    elif k == '1': cam_mode = 1
    elif k == '2': cam_mode = 2
    elif k == '3': cam_mode = 3

scene.bind('keydown', handle_key)

pause_btn  = button(text='  ⏸  PAUSE  ',  bind=toggle_pause_btn,
                    background=vector(0.0,0.18,0.30), color=vector(0.0,0.90,1.00))
replay_btn = button(text='  ↩  REPLAY  ', bind=do_replay,
                    background=vector(0.0,0.20,0.10), color=vector(0.15,1.0,0.55))
reset_btn  = button(text='  ⟳  RESET  ',  bind=do_reset,
                    background=vector(0.20,0.05,0.0), color=vector(1.0,0.55,0.10))

scene.append_to_caption(
    '<div style="display:inline-flex;align-items:center;gap:10px;'
    'background:linear-gradient(90deg,#002a3d,#003d5c);'
    'border:1px solid #0088aa;border-radius:6px;'
    'padding:5px 14px;margin:4px 0;font-family:monospace;font-size:12px;color:#70e8ff;">'
    '<span style="color:#a8dff5;letter-spacing:1px;">🏎️ SPEED</span>'
)
speed_display_lbl = wtext(text='  1.0×  ')
scene.append_to_caption('</div>')

def speed_slider_cb(s):
    global speed_mult
    speed_mult = round(s.value, 2)
    speed_display_lbl.text = f'  {speed_mult:.1f}×  '

spd_slider = slider(min=0.5, max=3.0, value=1.0, step=0.05,
                    length=260, width=12, left=8, bind=speed_slider_cb)
scene.append_to_caption(
    '  <span style="font-family:monospace;font-size:10px;color:#005577;">'
    '  0.5× slow ─── 3.0× fast &nbsp;&nbsp; CAM: [1] Follow  [2] Top  [3] FPV'
    '</span>\n'
)

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def fog_opacity(dist, max_dist=120.0, min_op=0.15):
    t = min(dist / max_dist, 1.0)
    return max(1.0 - t * (1.0 - min_op), min_op)

FLOOR_Y = SEA_FLOOR_Y

# ══════════════════════════════════════════════════════════════
#  OCEAN WATER VOLUME
# ══════════════════════════════════════════════════════════════
box(pos=vector(0, (FLOOR_Y + SEA_SURFACE_Y)/2, 0),
    size=vector(320, SEA_SURFACE_Y - FLOOR_Y, 140),
    color=vector(0.01, 0.22, 0.42),
    opacity=0.18)

# ══════════════════════════════════════════════════════════════
#  SEA FLOOR
# ══════════════════════════════════════════════════════════════
box(pos=vector(0, FLOOR_Y - 0.8, 0),
    size=vector(320, 1.8, 140),
    color=vector(0.62, 0.52, 0.32),
    texture=textures.rough)

random.seed(200)
for _ in range(18):
    rx = random.uniform(-130, 130)
    rz = random.uniform(-60, 60)
    box(pos=vector(rx, FLOOR_Y - 0.1, rz),
        size=vector(random.uniform(8, 25), 0.12, random.uniform(0.4, 1.2)),
        color=vector(0.55, 0.46, 0.28), opacity=0.55)

# ── Rocks ──────────────────────────────────────────────────
random.seed(42)
for _ in range(60):
    rx  = random.uniform(-120, 120)
    rz  = random.uniform(-55, 55)
    rs  = random.uniform(0.4, 3.5)
    rh  = random.uniform(0.35, 1.3)
    rtype = random.randint(0, 2)
    if rtype == 0:   rc = vector(random.uniform(0.40,0.60), random.uniform(0.32,0.48), random.uniform(0.18,0.28))
    elif rtype == 1: rc = vector(random.uniform(0.20,0.35), random.uniform(0.28,0.42), random.uniform(0.38,0.55))
    else:            rc = vector(random.uniform(0.55,0.75), random.uniform(0.28,0.40), random.uniform(0.20,0.32))
    d  = math.sqrt(rx*rx + rz*rz)
    op = fog_opacity(d, 130, 0.20)
    ellipsoid(pos=vector(rx, FLOOR_Y + rs*rh*0.3, rz),
              size=vector(rs*2.4, rs*rh, rs*1.8),
              color=rc, opacity=op)

# ── Coral formations ─────────────────────────────────────────
random.seed(303)
for _ in range(22):
    cx = random.uniform(-100, 100); cz = random.uniform(-45, 45)
    ch = random.uniform(1.5, 4.0)
    cc = vector(random.uniform(0.65,1.0), random.uniform(0.20,0.50), random.uniform(0.15,0.35))
    for branch in range(random.randint(2,4)):
        ba = random.uniform(0, 2*math.pi)
        cylinder(pos=vector(cx, FLOOR_Y, cz),
                 axis=vector(math.sin(ba)*ch*0.4, ch, math.cos(ba)*ch*0.3),
                 radius=random.uniform(0.08, 0.22), color=cc,
                 opacity=fog_opacity(math.sqrt(cx*cx+cz*cz), 130, 0.22))

# ── Starfish on seafloor (NEW) ────────────────────────────────
random.seed(444)
for _ in range(15):
    sx = random.uniform(-100, 100); sz = random.uniform(-45, 45)
    sr = random.uniform(0.5, 1.2)
    sc_col = vector(random.uniform(0.6,1.0), random.uniform(0.1,0.4), random.uniform(0.0,0.2))
    for arm in range(5):
        angle_arm = arm * 2*math.pi / 5
        cylinder(
            pos=vector(sx, FLOOR_Y + 0.05, sz),
            axis=vector(math.cos(angle_arm)*sr, 0.06, math.sin(angle_arm)*sr),
            radius=0.10, color=sc_col,
            opacity=fog_opacity(math.sqrt(sx*sx+sz*sz), 120, 0.25)
        )
    sphere(pos=vector(sx, FLOOR_Y+0.12, sz), radius=0.18, color=sc_col,
           opacity=fog_opacity(math.sqrt(sx*sx+sz*sz), 120, 0.25))

# ══════════════════════════════════════════════════════════════
#  ENHANCED SEAWEED / KELP (with frond tips)
# ══════════════════════════════════════════════════════════════
random.seed(91)
seaweed_list = []
seaweed_data = []
for _ in range(48):   # more stalks vs v16's 40
    kx   = random.uniform(-110, 110)
    kz   = random.uniform(-50, 50)
    kh   = random.uniform(3.0, 13.0)
    kph  = random.uniform(0, 2*math.pi)
    segs = random.randint(3, 6)
    seg_h= kh / segs
    ktype = random.randint(0,2)
    if ktype == 0:  base_col = vector(0.05, 0.72, 0.42)
    elif ktype==1:  base_col = vector(0.10, 0.55, 0.20)
    else:           base_col = vector(0.30, 0.65, 0.10)
    seg_objs = []
    for s in range(segs):
        y_bot = FLOOR_Y + s * seg_h
        brightness = 0.65 + 0.35 * s / segs
        cyl = cylinder(
            pos   = vector(kx, y_bot, kz),
            axis  = vector(0, seg_h, 0),
            radius= random.uniform(0.06, 0.18),
            color = vector(base_col.x*brightness, base_col.y*brightness, base_col.z*brightness),
            opacity=fog_opacity(math.sqrt(kx*kx+kz*kz), 130, 0.20)
        )
        seg_objs.append(cyl)
    # Frond tip (leaf-like ellipsoid at top)
    tip_col = vector(min(base_col.x*1.3,1), min(base_col.y*1.3,1), min(base_col.z*1.3,1))
    frond = ellipsoid(
        pos=vector(kx, FLOOR_Y + kh + 0.5, kz),
        size=vector(0.4, 1.2, 0.15),
        color=tip_col,
        opacity=fog_opacity(math.sqrt(kx*kx+kz*kz), 130, 0.22)
    )
    seg_objs.append(frond)
    seaweed_list.append(seg_objs)
    seaweed_data.append({'x':kx,'z':kz,'h':kh,'ph':kph,'segs':segs,'seg_h':seg_h})

def update_seaweed(t_sim):
    for seg_objs, sd in zip(seaweed_list, seaweed_data):
        for s, obj in enumerate(seg_objs):
            if s < sd['segs']:   # cylinder segments
                sway_x = 0.22 * (s+1) * math.sin(t_sim*0.75 + sd['ph'] + s*0.5)
                sway_z = 0.14 * (s+1) * math.cos(t_sim*0.55 + sd['ph'] + s*0.3)
                obj.axis = vector(sway_x, sd['seg_h'], sway_z)
            else:  # frond tip
                sway_x = 0.35 * math.sin(t_sim*0.80 + sd['ph'])
                sway_z = 0.25 * math.cos(t_sim*0.60 + sd['ph'])
                obj.pos = vector(sd['x'] + sway_x, FLOOR_Y + sd['h'] + 0.5, sd['z'] + sway_z)

# ══════════════════════════════════════════════════════════════
#  BUBBLES — original stream + 3 seafloor vents (NEW)
# ══════════════════════════════════════════════════════════════
random.seed(77)
bubbles = []
for _ in range(150):
    bx  = random.uniform(-110, 110)
    bz  = random.uniform(-52, 52)
    by  = random.uniform(FLOOR_Y, FLOOR_Y + 22)
    br  = random.uniform(0.05, 0.30)
    bspd= random.uniform(0.08, 0.24)
    b   = sphere(pos=vector(bx, by, bz), radius=br,
                 color=vector(0.45, 0.88, 0.95),
                 opacity=random.uniform(0.06, 0.26))
    bubbles.append({'o':b, 'spd':bspd,
                    'dx':random.uniform(-0.03,0.03),
                    'dz':random.uniform(-0.02,0.02),
                    'ox':bx, 'oz':bz})

# Seafloor vent bubble streams (3 vents, denser + bigger)
VENT_POSITIONS = [(-40, -18), (10, 25), (55, -10)]
vent_bubbles = []
random.seed(555)
for vx, vz in VENT_POSITIONS:
    for _ in range(30):
        br  = random.uniform(0.08, 0.45)
        bspd= random.uniform(0.18, 0.40)
        by  = random.uniform(FLOOR_Y, FLOOR_Y + 15)
        b   = sphere(pos=vector(vx + random.uniform(-0.8,0.8),
                                by,
                                vz + random.uniform(-0.8,0.8)),
                     radius=br,
                     color=vector(0.70, 0.95, 1.00),
                     opacity=random.uniform(0.15, 0.45))
        vent_bubbles.append({'o':b, 'spd':bspd,
                             'ox':vx, 'oz':vz,
                             'dx':random.uniform(-0.015,0.015),
                             'dz':random.uniform(-0.015,0.015)})

def update_bubbles(t_sim):
    for b in bubbles:
        b['o'].pos.y += b['spd'] * DT * 22
        b['o'].pos.x += b['dx']
        b['o'].pos.z += b['dz']
        if b['o'].pos.y > SEA_SURFACE_Y + 5:
            b['o'].pos.y = FLOOR_Y + random.uniform(0, 3)
            b['o'].pos.x = random.uniform(-110, 110)
            b['o'].pos.z = random.uniform(-52, 52)
    for b in vent_bubbles:
        b['o'].pos.y += b['spd'] * DT * 26
        b['o'].pos.x += b['dx']
        b['o'].pos.z += b['dz']
        # slight wobble
        b['o'].pos.x += 0.008 * math.sin(t_sim * 3.0 + b['spd'] * 10)
        if b['o'].pos.y > SEA_SURFACE_Y + 3:
            b['o'].pos.y = FLOOR_Y + random.uniform(0, 2)
            b['o'].pos.x = b['ox'] + random.uniform(-0.8, 0.8)
            b['o'].pos.z = b['oz'] + random.uniform(-0.8, 0.8)

# ══════════════════════════════════════════════════════════════
#  🐟  FISH SCHOOLS (NEW)
#  5 schools, 8 fish each — body + tail + fins, animated flocking
# ══════════════════════════════════════════════════════════════
random.seed(888)

FISH_COLORS = [
    (vector(0.95, 0.55, 0.05), vector(1.00, 0.80, 0.10)),  # orange/gold
    (vector(0.10, 0.50, 0.90), vector(0.40, 0.85, 1.00)),  # blue/cyan
    (vector(0.70, 0.10, 0.60), vector(0.90, 0.30, 0.80)),  # purple/pink
    (vector(0.10, 0.75, 0.40), vector(0.30, 1.00, 0.60)),  # green/teal
    (vector(0.85, 0.20, 0.15), vector(1.00, 0.50, 0.30)),  # red/orange
]

fish_schools = []

for school_idx in range(5):
    body_col, fin_col = FISH_COLORS[school_idx % len(FISH_COLORS)]
    # School center - spread across FULL scene including pipe height and above
    cx = random.uniform(-80, 80)
    cy = random.uniform(FLOOR_Y + 8, VERT_PIPE_TOP + 10)   # full vertical range
    cz = random.uniform(-45, 45)
    school_speed = random.uniform(0.06, 0.14)   # slightly faster so they roam more
    school_dir   = norm(vector(random.uniform(-1,1), random.uniform(-0.15,0.15), random.uniform(-1,1)))
    school_phase = random.uniform(0, 2*math.pi)

    fish_list = []
    for fi in range(8):
        # Offset each fish within the school
        ox = random.uniform(-5, 5)
        oy = random.uniform(-2.5, 2.5)
        oz = random.uniform(-5, 5)
        fpos = vector(cx+ox, cy+oy, cz+oz)
        fscale = random.uniform(0.55, 1.0)

        # Fish body
        body  = ellipsoid(pos=fpos, size=vector(fscale*2.0, fscale*0.65, fscale*0.55),
                          color=body_col, opacity=0.92)
        # Tail fin
        tail  = pyramid(pos=fpos - vector(fscale, 0, 0),
                        size=vector(fscale*0.90, fscale*0.80, fscale*0.12),
                        color=fin_col, opacity=0.88)
        # Dorsal fin
        dors  = pyramid(pos=fpos + vector(fscale*0.1, fscale*0.35, 0),
                        size=vector(fscale*0.5, fscale*0.45, fscale*0.08),
                        color=fin_col, opacity=0.80)
        # Eye
        eye   = sphere(pos=fpos + vector(fscale*0.75, fscale*0.10, fscale*0.24),
                       radius=fscale*0.09, color=vector(0.02,0.02,0.02))

        fish_list.append({
            'body': body, 'tail': tail, 'dors': dors, 'eye': eye,
            'ox': ox, 'oy': oy, 'oz': oz,
            'scale': fscale, 'phase': random.uniform(0, 2*math.pi)
        })

    fish_schools.append({
        'fish': fish_list,
        'cx': cx, 'cy': cy, 'cz': cz,
        'dir': school_dir,
        'speed': school_speed,
        'phase': school_phase,
        'turn_timer': random.uniform(0, 8),
        'body_col': body_col, 'fin_col': fin_col
    })


def update_fish(t_sim):
    for school in fish_schools:
        # Slowly steer school: random gentle turns
        school['turn_timer'] -= DT
        if school['turn_timer'] <= 0:
            # Pick a new gentle direction
            turn_x = random.uniform(-0.4, 0.4)
            turn_y = random.uniform(-0.05, 0.05)
            turn_z = random.uniform(-0.4, 0.4)
            new_dir = norm(school['dir'] + vector(turn_x, turn_y, turn_z))
            school['dir'] = new_dir
            school['turn_timer'] = random.uniform(3, 10)

        # Boundary bounce — keep fish in the scene
        spd = school['speed']
        school['cx'] += school['dir'].x * spd
        school['cy'] += school['dir'].y * spd
        school['cz'] += school['dir'].z * spd

        if school['cx'] >  100: school['dir'].x = -abs(school['dir'].x)
        if school['cx'] < -100: school['dir'].x =  abs(school['dir'].x)
        # Full vertical range — fish roam everywhere including at/above pipeline
        if school['cy'] > VERT_PIPE_TOP + 12: school['dir'].y = -abs(school['dir'].y)
        if school['cy'] < FLOOR_Y + 4:        school['dir'].y =  abs(school['dir'].y)
        if school['cz'] >  55: school['dir'].z = -abs(school['dir'].z)
        if school['cz'] < -55: school['dir'].z =  abs(school['dir'].z)
        # Occasional random vertical nudge so fish visit all depths
        if random.random() < 0.004:
            school['dir'] = norm(school['dir'] + vector(0, random.uniform(-0.35, 0.35), 0))

        # Compute heading angle for body rotation
        hd = school['dir']
        yaw = math.atan2(hd.z, hd.x)

        for fi, f in enumerate(school['fish']):
            # Each fish orbits slightly around school center + undulates
            fi_phase = f['phase'] + t_sim * 1.8
            ox_r = f['ox'] * math.cos(t_sim*0.15) - f['oz'] * math.sin(t_sim*0.15)
            oz_r = f['ox'] * math.sin(t_sim*0.15) + f['oz'] * math.cos(t_sim*0.15)
            oy_r = f['oy'] + 0.5 * math.sin(fi_phase)

            fpos = vector(school['cx'] + ox_r,
                          school['cy'] + oy_r,
                          school['cz'] + oz_r)
            sc = f['scale']

            # Orient fish along heading direction
            fwd = vector(math.cos(yaw), 0, math.sin(yaw))
            up  = vector(0, 1, 0)
            side= norm(cross(fwd, up))

            # Body position
            f['body'].pos  = fpos
            f['body'].axis = fwd * sc * 2.0
            f['body'].up   = up

            # Tail wag
            wag = 0.35 * math.sin(t_sim * 4.5 + f['phase'])
            tail_pos = fpos - fwd * sc
            f['tail'].pos  = tail_pos
            f['tail'].axis = (-fwd + side * wag) * sc * 0.9
            f['tail'].up   = up

            # Dorsal fin
            f['dors'].pos  = fpos + up * sc * 0.35 + fwd * sc * 0.1
            f['dors'].axis = fwd * sc * 0.5
            f['dors'].up   = up

            # Eye
            f['eye'].pos = fpos + fwd * sc * 0.75 + up * sc * 0.10 + side * sc * 0.24


# ══════════════════════════════════════════════════════════════
#  🪼  JELLYFISH (NEW) — pulsing bell + trailing tentacles
# ══════════════════════════════════════════════════════════════
random.seed(321)
jellyfish_list = []

JELLY_COLORS = [
    vector(0.80, 0.30, 1.00),  # purple
    vector(0.30, 0.80, 1.00),  # cyan
    vector(1.00, 0.55, 0.80),  # pink
    vector(0.50, 1.00, 0.75),  # teal
]

for ji in range(12):
    jx = random.uniform(-95, 95)
    jy = random.uniform(FLOOR_Y + 5, PIPE_Y - 15)
    jz = random.uniform(-45, 45)
    jcol = JELLY_COLORS[ji % len(JELLY_COLORS)]
    jscale = random.uniform(0.6, 1.8)
    jphase = random.uniform(0, 2*math.pi)
    jspd_y = random.uniform(0.008, 0.025)  # drift speed
    jdx    = random.uniform(-0.012, 0.012)
    jdz    = random.uniform(-0.012, 0.012)

    # Bell (hemisphere-like: flattened sphere)
    bell = sphere(pos=vector(jx, jy, jz),
                  radius=jscale * 1.0,
                  color=jcol, opacity=0.55)

    # Inner glow (slightly smaller, brighter, more opaque)
    glow = sphere(pos=vector(jx, jy - jscale*0.2, jz),
                  radius=jscale * 0.55,
                  color=vector(min(jcol.x*1.4,1), min(jcol.y*1.4,1), min(jcol.z*1.4,1)),
                  emissive=True, opacity=0.35)

    # Tentacles (4-6 thin cylinders hanging below bell)
    tentacles = []
    n_tent = random.randint(4, 7)
    for t in range(n_tent):
        ta = t * 2*math.pi / n_tent
        tr_x = math.cos(ta) * jscale * 0.5
        tr_z = math.sin(ta) * jscale * 0.5
        t_len = random.uniform(1.8, 4.5) * jscale
        tent = cylinder(
            pos=vector(jx + tr_x, jy - jscale*0.5, jz + tr_z),
            axis=vector(tr_x*0.15, -t_len, tr_z*0.15),
            radius=0.05 * jscale,
            color=jcol, opacity=0.40
        )
        tentacles.append({'obj': tent, 'tx': tr_x, 'tz': tr_z,
                          'len': t_len, 'angle': ta})

    jellyfish_list.append({
        'bell': bell, 'glow': glow, 'tentacles': tentacles,
        'x': jx, 'y': jy, 'z': jz,
        'scale': jscale, 'phase': jphase,
        'spd_y': jspd_y, 'dx': jdx, 'dz': jdz,
        'col': jcol, 'dir_y': 1.0
    })


def update_jellyfish(t_sim):
    for j in jellyfish_list:
        # Pulsing bell: scale oscillates
        pulse = 0.85 + 0.20 * abs(math.sin(t_sim * 1.2 + j['phase']))
        bell_r = j['scale'] * pulse
        j['bell'].radius = bell_r
        j['glow'].radius  = j['scale'] * 0.55 * pulse

        # Drift upward then sink gently (yo-yo)
        j['y'] += j['spd_y'] * j['dir_y']
        j['x'] += j['dx'] + 0.005 * math.sin(t_sim * 0.6 + j['phase'])
        j['z'] += j['dz'] + 0.005 * math.cos(t_sim * 0.5 + j['phase'])

        # Reverse direction at bounds
        if j['y'] > PIPE_Y - 8:  j['dir_y'] = -1.0
        if j['y'] < FLOOR_Y + 4: j['dir_y'] =  1.0
        if j['x'] >  95: j['dx'] = -abs(j['dx'])
        if j['x'] < -95: j['dx'] =  abs(j['dx'])
        if j['z'] >  48: j['dz'] = -abs(j['dz'])
        if j['z'] < -48: j['dz'] =  abs(j['dz'])

        jpos = vector(j['x'], j['y'], j['z'])
        j['bell'].pos = jpos
        j['glow'].pos = jpos - vector(0, j['scale']*0.2, 0)

        # Glow pulse intensity
        gp = 0.20 + 0.25 * abs(math.sin(t_sim * 1.2 + j['phase']))
        j['glow'].opacity = gp

        # Animate tentacles (wave)
        for ti, t in enumerate(j['tentacles']):
            wave_x = 0.30 * math.sin(t_sim * 2.0 + j['phase'] + ti * 0.8)
            wave_z = 0.30 * math.cos(t_sim * 1.8 + j['phase'] + ti * 0.7)
            t_len = t['len'] * (0.9 + 0.15 * math.sin(t_sim * 1.2 + j['phase']))
            t['obj'].pos = (jpos + vector(t['tx'], -j['scale']*0.5*pulse, t['tz']))
            t['obj'].axis = vector(t['tx']*0.15 + wave_x, -t_len, t['tz']*0.15 + wave_z)


# ══════════════════════════════════════════════════════════════
#  BIOLUMINESCENT PLANKTON (NEW) — tiny glowing drifting motes
# ══════════════════════════════════════════════════════════════
random.seed(999)
plankton = []
PLANKTON_COLORS = [
    vector(0.20, 1.00, 0.70),  # green
    vector(0.40, 0.60, 1.00),  # blue
    vector(0.80, 0.20, 1.00),  # violet
]
for _ in range(200):
    px = random.uniform(-100, 100)
    py = random.uniform(FLOOR_Y + 1, PIPE_Y - 5)
    pz = random.uniform(-50, 50)
    pc = PLANKTON_COLORS[random.randint(0, len(PLANKTON_COLORS)-1)]
    pp = random.uniform(0, 2*math.pi)
    pspd = random.uniform(0.003, 0.012)
    p = sphere(pos=vector(px, py, pz),
               radius=random.uniform(0.06, 0.14),
               color=pc, emissive=True,
               opacity=random.uniform(0.05, 0.25))
    plankton.append({'o': p, 'x': px, 'y': py, 'z': pz,
                     'phase': pp, 'spd': pspd,
                     'dx': random.uniform(-0.008, 0.008),
                     'dz': random.uniform(-0.008, 0.008)})

def update_plankton(t_sim):
    for p in plankton:
        p['y'] += p['spd'] * math.sin(t_sim * 0.3 + p['phase']) * 0.1
        p['x'] += p['dx']
        p['z'] += p['dz']
        # Gentle blink
        p['o'].opacity = 0.08 + 0.18 * abs(math.sin(t_sim * 2.5 + p['phase']))
        p['o'].pos = vector(p['x'], p['y'], p['z'])
        # Wrap around edges
        if p['x'] >  102: p['x'] = -100
        if p['x'] < -102: p['x'] =  100
        if p['z'] >  52:  p['z'] = -50
        if p['z'] < -52:  p['z'] =  50


# Sea surface shimmer
box(pos=vector(0, SEA_SURFACE_Y - 0.3, 0),
    size=vector(320, 0.6, 140),
    color=vector(0.10, 0.55, 0.75),
    opacity=0.30)

# ══════════════════════════════════════════════════════════════
#  PIPELINE NODES
# ══════════════════════════════════════════════════════════════
N_HORIZ_SEGS = 6
horiz_x_vals = [HORIZ_PIPE_X0 + i*(HORIZ_PIPE_X1-HORIZ_PIPE_X0)/N_HORIZ_SEGS
                for i in range(N_HORIZ_SEGS+1)]
horiz_z_vals = [HORIZ_ZIG * ((-1)**i) for i in range(N_HORIZ_SEGS+1)]
horiz_z_vals[0] = 0.0; horiz_z_vals[-1] = 0.0
pipe_nodes = [vector(horiz_x_vals[i], PIPE_Y, horiz_z_vals[i])
              for i in range(N_HORIZ_SEGS+1)]

N_VERT_SEGS = 5
vert_pipe_nodes = [
    vector(VERT_PIPE_X,
           VERT_PIPE_BOT + i*(VERT_PIPE_TOP-VERT_PIPE_BOT)/N_VERT_SEGS,
           VERT_PIPE_Z)
    for i in range(N_VERT_SEGS+1)
]

# ══════════════════════════════════════════════════════════════
#  PIPE DRAWING
# ══════════════════════════════════════════════════════════════
def draw_pipe_segment(p1, p2, is_vertical=False):
    ax  = p2 - p1
    slen= mag(ax)
    cylinder(pos=p1, axis=ax, radius=PIPE_RADIUS,
             color=vector(0.38,0.38,0.45), texture=textures.metal)
    ring(pos=p1,        axis=ax, radius=PIPE_RADIUS*1.35, thickness=0.32, color=vector(0.25,0.25,0.29))
    ring(pos=p1+ax*0.5, axis=ax, radius=PIPE_RADIUS*1.20, thickness=0.24, color=vector(0.27,0.27,0.31))
    ring(pos=p2,        axis=ax, radius=PIPE_RADIUS*1.35, thickness=0.32, color=vector(0.25,0.25,0.29))
    if not is_vertical:
        nleg = max(1, int(slen // 30))
        for k in range(nleg):
            t  = (k + 0.5) / nleg
            lp = p1 + ax * t
            cylinder(pos=lp, axis=vector(0, FLOOR_Y - lp.y, 0),
                     radius=0.12, color=vector(0.22,0.22,0.26), opacity=0.40)
    else:
        nbrk = max(1, int(slen // 15))
        for k in range(nbrk):
            t  = (k + 0.5) / nbrk
            bp = p1 + ax * t
            for side in [vector(0, 0, 2.5), vector(0, 0, -2.5)]:
                cylinder(pos=bp, axis=side, radius=0.14, color=vector(0.22,0.22,0.26))

for i in range(len(pipe_nodes)-1):
    draw_pipe_segment(pipe_nodes[i], pipe_nodes[i+1])
for nd in pipe_nodes[1:-1]:
    sphere(pos=nd, radius=PIPE_RADIUS*1.40, color=vector(0.33,0.33,0.39), texture=textures.metal)

sphere(pos=vert_pipe_nodes[0], radius=PIPE_RADIUS*1.65,
       color=vector(0.35,0.35,0.42), texture=textures.metal)
ring(pos=vert_pipe_nodes[0], axis=vector(0,1,0),
     radius=PIPE_RADIUS*1.85, thickness=0.45,
     color=vector(0.10,0.80,1.0), opacity=0.75)

for i in range(len(vert_pipe_nodes)-1):
    draw_pipe_segment(vert_pipe_nodes[i], vert_pipe_nodes[i+1], is_vertical=True)
for nd in vert_pipe_nodes[1:-1]:
    sphere(pos=nd, radius=PIPE_RADIUS*1.35, color=vector(0.33,0.33,0.39), texture=textures.metal)

cylinder(pos=vert_pipe_nodes[-1], axis=vector(0, 2.5, 0),
         radius=PIPE_RADIUS*0.85, color=vector(0.26,0.26,0.32))
ring(pos=vert_pipe_nodes[-1]+vector(0,2.5,0), axis=vector(0,1,0),
     radius=PIPE_RADIUS*1.25, thickness=0.55, color=vector(0.30,0.30,0.35))

# ══════════════════════════════════════════════════════════════
#  PIPELINE TEMPERATURE RINGS
# ══════════════════════════════════════════════════════════════
pipe_temp_rings = []
vert_temp_rings = []

for i in range(len(pipe_nodes)-1):
    p1  = pipe_nodes[i]; p2 = pipe_nodes[i+1]
    mid = p1 + (p2-p1)*0.50
    ax  = norm(p2-p1)
    t_c = PIPE_SEG_TEMPS[i]
    col = pipe_temp_to_color(t_c)
    ri  = ring(pos=mid, axis=ax, radius=PIPE_RADIUS*1.60, thickness=0.30,
               color=col, emissive=True, opacity=0.95)
    rg  = ring(pos=mid, axis=ax, radius=PIPE_RADIUS*2.60, thickness=0.60,
               color=col, emissive=True, opacity=0.18)
    ls  = sphere(pos=mid+vector(0, PIPE_RADIUS*2.2, 0), radius=0.35,
                 color=col, emissive=True, opacity=0.85)
    pipe_temp_rings.append({'inner':ri,'glow':rg,'lbl':ls,
                            'mid':mid,'ax':ax,'temp':t_c,'seg':i,
                            'base_col':col,'scanned':False})

for i in range(len(vert_pipe_nodes)-1):
    p1  = vert_pipe_nodes[i]; p2 = vert_pipe_nodes[i+1]
    mid = p1 + (p2-p1)*0.50
    ax  = norm(p2-p1)
    t_c = VERT_SEG_TEMPS[i]
    col = pipe_temp_to_color(t_c)
    ri  = ring(pos=mid, axis=ax, radius=PIPE_RADIUS*1.60, thickness=0.30,
               color=col, emissive=True, opacity=0.95)
    rg  = ring(pos=mid, axis=ax, radius=PIPE_RADIUS*2.60, thickness=0.60,
               color=col, emissive=True, opacity=0.18)
    ls  = sphere(pos=mid+vector(PIPE_RADIUS*2.2, 0, 0), radius=0.35,
                 color=col, emissive=True, opacity=0.85)
    vert_temp_rings.append({'inner':ri,'glow':rg,'lbl':ls,
                            'mid':mid,'ax':ax,'temp':t_c,'seg':i,
                            'base_col':col,'scanned':False})

all_temp_rings = pipe_temp_rings + vert_temp_rings

# ══════════════════════════════════════════════════════════════
#  CORROSION SPOTS
# ══════════════════════════════════════════════════════════════
SEVERITY_DEFS = [
    ('minor',    vector(1.00,0.95,0.10), 0.40),
    ('moderate', vector(0.95,0.45,0.05), 0.55),
    ('severe',   vector(0.90,0.08,0.04), 0.70),
]
random.seed(7)
corrosion_spots=[]; corrosion_markers=[]; corrosion_severity=[]; corrosion_colors=[]
for i in range(len(pipe_nodes)-1):
    p1,p2 = pipe_nodes[i], pipe_nodes[i+1]
    cp = p1 + (p2-p1)*0.55 + vector(0, PIPE_RADIUS, 0)
    corrosion_spots.append(cp)
    si = random.choice([0,0,1,1,2]); sn,sc,sr = SEVERITY_DEFS[si]
    corrosion_severity.append(sn); corrosion_colors.append(sc)
    corrosion_markers.append(sphere(pos=cp, radius=sr, color=sc, emissive=True, opacity=0.92))

random.seed(17)
vert_corr_spots=[]; vert_corr_markers=[]; vert_corr_severity=[]; vert_corr_colors=[]
for i in range(len(vert_pipe_nodes)-1):
    p1,p2 = vert_pipe_nodes[i], vert_pipe_nodes[i+1]
    cp = p1 + (p2-p1)*0.60 + vector(PIPE_RADIUS, 0, 0)
    vert_corr_spots.append(cp)
    si = random.choice([0,0,1,1,2]); sn,sc,sr = SEVERITY_DEFS[si]
    vert_corr_severity.append(sn); vert_corr_colors.append(sc)
    vert_corr_markers.append(sphere(pos=cp, radius=sr, color=sc, emissive=True, opacity=0.92))

# ══════════════════════════════════════════════════════════════
#  CRACKS
# ══════════════════════════════════════════════════════════════
random.seed(13)
CRACK_SEV_LIST = ['hairline','moderate','fracture']
crack_col_map  = {'hairline':vector(0.90,0.55,0.05),
                  'moderate':vector(0.95,0.20,0.02),
                  'fracture':vector(1.00,0.02,0.02)}
CRACK_SCANNED_COL = vector(0.05,1.0,0.30)
crack_positions=[]; crack_markers=[]; crack_scars=[]; crack_severities=[]
for i in range(len(pipe_nodes)-1):
    p1,p2 = pipe_nodes[i], pipe_nodes[i+1]
    cp = p1 + (p2-p1)*0.28; crack_p = cp + vector(0, PIPE_RADIUS, 0)
    crack_positions.append(crack_p)
    sev = random.choice(CRACK_SEV_LIST); crack_severities.append(sev)
    col = crack_col_map[sev]; clen = random.uniform(2.5, 5.0)
    cw  = 0.55 if sev=='hairline' else (0.90 if sev=='moderate' else 1.40)
    ch  = 0.30 if sev=='hairline' else (0.55 if sev=='moderate' else 0.90)
    crack_scars.append(box(pos=crack_p, axis=norm(p2-p1)*clen,
                           size=vector(clen,ch,cw), color=vector(0.04,0,0), opacity=0.98))
    rad = 0.70 if sev=='hairline' else (1.00 if sev=='moderate' else 1.40)
    crack_markers.append(sphere(pos=crack_p+vector(0,0.90,0), radius=rad,
                                color=col, emissive=True, opacity=0.95))

random.seed(23)
vert_crack_positions=[]; vert_crack_markers=[]; vert_crack_scars=[]; vert_crack_severities=[]
for i in range(len(vert_pipe_nodes)-1):
    p1,p2 = vert_pipe_nodes[i], vert_pipe_nodes[i+1]
    cp = p1 + (p2-p1)*0.38; crack_p = cp + vector(PIPE_RADIUS, 0, 0)
    vert_crack_positions.append(crack_p)
    sev = random.choice(CRACK_SEV_LIST); vert_crack_severities.append(sev)
    col = crack_col_map[sev]; clen = random.uniform(2.0, 4.0)
    cw  = 0.55 if sev=='hairline' else (0.90 if sev=='moderate' else 1.30)
    ch  = 0.30 if sev=='hairline' else (0.50 if sev=='moderate' else 0.85)
    vert_crack_scars.append(box(pos=crack_p, axis=vector(0,1,0)*clen,
                                size=vector(ch,clen,cw), color=vector(0.04,0,0), opacity=0.98))
    rad = 0.65 if sev=='hairline' else (0.95 if sev=='moderate' else 1.35)
    vert_crack_markers.append(sphere(pos=crack_p+vector(0.85,0,0), radius=rad,
                                     color=col, emissive=True, opacity=0.95))

# ══════════════════════════════════════════════════════════════
#  CURRENT ARROWS
# ══════════════════════════════════════════════════════════════
current_arrows = []
random.seed(55)
for zx,zz,zr,zs,zd in CURRENT_ZONES:
    nd_ = norm(vector(zd.x, 0, zd.z))
    for _ in range(6):
        ax_ = random.uniform(zx-zr*0.7, zx+zr*0.7)
        az_ = random.uniform(zz-zr*0.7, zz+zr*0.7)
        arr = arrow(pos=vector(ax_, PIPE_Y+1.0, az_), axis=nd_*4.0,
                    shaftwidth=0.20, color=vector(0.10,0.55,1.0), opacity=0.28)
        current_arrows.append({'obj':arr,'dir':nd_,'str':zs})

random.seed(77); vert_cur_arrows = []
_,_,_,_,vzd = VERT_CURRENT_ZONE
for _ in range(8):
    ay_ = random.uniform(VERT_PIPE_BOT+5, VERT_PIPE_TOP-5)
    nd_ = norm(vector(vzd.x, vzd.y, vzd.z))
    arr = arrow(pos=vector(VERT_PIPE_X+3, ay_, VERT_PIPE_Z+random.uniform(-3,3)),
                axis=nd_*3.5, shaftwidth=0.18,
                color=vector(0.20,0.75,1.0), opacity=0.30)
    vert_cur_arrows.append({'obj':arr,'dir':nd_})

# ══════════════════════════════════════════════════════════════
#  OBSTACLES
# ══════════════════════════════════════════════════════════════
BUOY_Y = PIPE_Y + 3.0
buoy_defs = [
    (-55,  0.0), (-30, -8.0), (-5,  9.0), (20,  -6.0),
    ( 45,  7.0), ( 60, -4.0), (-42, 11.0), (10,  -9.0),
    ( 35, -5.0), (-18,  8.0),
]
obstacle_positions = []

def draw_buoy(pos, tether_bot=None):
    sc = 1.0
    sphere(pos=pos+vector(0,sc*1.2,0), radius=sc*1.35,
           color=vector(0.95,0.82,0.04), texture=textures.metal)
    cone(pos=pos+vector(0,-sc*0.4,0), axis=vector(0,sc*1.6,0),
         radius=sc*1.3, color=vector(0.90,0.76,0.03))
    cylinder(pos=pos+vector(0,-sc*2.0,0), axis=vector(0,sc*1.6,0),
             radius=sc*0.25, color=vector(0.55,0.52,0.35))
    for wr in [0.4, 0.9, 1.4]:
        ring(pos=pos+vector(0,wr*sc,0), axis=vector(0,1,0),
             radius=sc*1.38, thickness=sc*0.18, color=vector(0.12,0.10,0.06))
    sphere(pos=pos+vector(0,sc*2.7,0), radius=sc*0.30,
           color=vector(1.0,0.15,0.0), emissive=True)
    if tether_bot is not None:
        cylinder(pos=pos+vector(0,-sc*2.15,0),
                 axis=vector(0, tether_bot - (pos.y-sc*2.15), 0),
                 radius=0.07, color=vector(0.40,0.38,0.26), opacity=0.35)

for bx, bz in buoy_defs:
    op = vector(bx, BUOY_Y, bz)
    obstacle_positions.append(op)
    draw_buoy(op, tether_bot=FLOOR_Y)

vert_obstacle_defs = [
    (VERT_PIPE_X+6, VERT_PIPE_BOT+10, VERT_PIPE_Z+4),
    (VERT_PIPE_X-6, VERT_PIPE_BOT+30, VERT_PIPE_Z-3),
    (VERT_PIPE_X+5, VERT_PIPE_BOT+55, VERT_PIPE_Z+5),
    (VERT_PIPE_X-5, VERT_PIPE_BOT+75, VERT_PIPE_Z-4),
]
vert_obstacle_positions = []
for ox,oy,oz in vert_obstacle_defs:
    op = vector(ox, oy, oz)
    vert_obstacle_positions.append(op)
    draw_buoy(op, tether_bot=FLOOR_Y)

all_obstacle_positions = obstacle_positions + vert_obstacle_positions

# ══════════════════════════════════════════════════════════════
#  WAYPOINTS
# ══════════════════════════════════════════════════════════════
def make_helix_waypoints():
    wps = []; ga = 0.0
    for si in range(len(pipe_nodes)-1):
        p1 = pipe_nodes[si]; p2 = pipe_nodes[si+1]
        ax = norm(p2-p1)
        wu = vector(0,1,0)
        if abs(dot(ax,wu))>0.9: wu = vector(1,0,0)
        pp1 = norm(cross(wu,ax)); pp2 = norm(cross(ax,pp1))
        for k in range(HELIX_STEPS):
            t   = k / HELIX_STEPS
            c   = p1 + (p2-p1)*t
            ang = ga + t*(2.0*math.pi*HELIX_TURNS)
            wps.append(c + pp1*math.cos(ang)*HELIX_RADIUS
                         + pp2*math.sin(ang)*HELIX_RADIUS)
        ga += 2.0*math.pi*HELIX_TURNS
    last = pipe_nodes[-1]
    wps.append(vector(last.x, last.y+HELIX_RADIUS*0.5, last.z))
    return wps

def make_vert_helix_waypoints():
    wps = []; ga = 0.0
    for si in range(len(vert_pipe_nodes)-1):
        p1 = vert_pipe_nodes[si]; p2 = vert_pipe_nodes[si+1]
        for k in range(VERT_HELIX_STEPS):
            t   = k / VERT_HELIX_STEPS
            c   = p1 + (p2-p1)*t
            ang = ga + t*(2.0*math.pi*VERT_HELIX_TURNS)
            wps.append(c + vector(VERT_HELIX_RADIUS*math.cos(ang),
                                  0,
                                  VERT_HELIX_RADIUS*math.sin(ang)))
        ga += 2.0*math.pi*VERT_HELIX_TURNS
    top = vert_pipe_nodes[-1]
    wps.append(top + vector(0, 3.0, 0))
    return wps

horiz_waypoints = make_helix_waypoints()
vert_waypoints  = make_vert_helix_waypoints()

TRANS_STEPS = 40; transition_wps = []
h_end = horiz_waypoints[-1]; v_start = vert_waypoints[0]
for k in range(1, TRANS_STEPS+1):
    t = k / TRANS_STEPS
    transition_wps.append(h_end + (v_start - h_end)*t)

waypoints    = horiz_waypoints + transition_wps + vert_waypoints
HORIZ_WP_END = len(horiz_waypoints) + len(transition_wps)
sp = vector(waypoints[0].x, waypoints[0].y, waypoints[0].z)

# ══════════════════════════════════════════════════════════════
#  DRONE
# ══════════════════════════════════════════════════════════════
drone_body  = box(pos=sp, size=vector(2.6,0.90,1.30),
                  color=vector(0.10,0.58,0.88), texture=textures.metal)
drone_nose  = cone(pos=sp+vector(1.3,0,0), axis=vector(1.0,0,0),
                   radius=0.45, color=vector(0.07,0.42,0.70))
drone_cam   = sphere(pos=sp+vector(1.38,0,0), radius=0.28,
                     color=vector(0.03,0.03,0.07))
drone_dome  = sphere(pos=sp+vector(0,0.62,0), radius=0.38,
                     color=vector(0.22,0.90,1.0), opacity=0.70)

drone_ll  = sphere(pos=sp+vector(1.1,0.14,0.35), radius=0.14,
                   color=vector(0.05,1.0,0.20), emissive=True)
drone_lr  = sphere(pos=sp+vector(1.1,0.14,-0.35), radius=0.14,
                   color=vector(1.0,0.10,0.10), emissive=True)
drone_beam= cone(pos=sp+vector(1.0,0,0), axis=vector(7.0,0,0),
                 radius=2.8, color=vector(1.0,0.97,0.62), opacity=0.06)
drone_shad= ring(pos=vector(sp.x,PIPE_Y+PIPE_RADIUS+0.15,sp.z),
                 axis=vector(0,1,0), radius=0.90,
                 thickness=0.08, color=vector(0,0,0), opacity=0.28)

drone_beacon = sphere(pos=sp+vector(0,-0.50,0), radius=0.22,
                      color=vector(0.15,0.50,1.0), emissive=True, opacity=0.85)

thr_off = [vector(-0.70,0.52, 0.70),
           vector( 0.70,0.52, 0.70),
           vector(-0.70,0.52,-0.70),
           vector( 0.70,0.52,-0.70)]
thrusters = []; props = []; thr_glows = []
for tf in thr_off:
    thrusters.append(ring(pos=sp+tf, axis=vector(0,1,0),
                          radius=0.36, thickness=0.09, color=vector(0.36,0.36,0.42)))
    props.append(box(pos=sp+tf+vector(0,0.02,0),
                     size=vector(0.06,0.06,0.70), color=vector(0.65,0.65,0.70)))
    thr_glows.append(ring(pos=sp+tf+vector(0,-0.05,0), axis=vector(0,1,0),
                          radius=0.44, thickness=0.10,
                          color=vector(0.0,0.90,1.0), emissive=True, opacity=0.30))

drone_spotlight  = local_light(pos=sp+vector(3,0,0), color=vector(0.95,0.92,0.80))
drone_port_light = local_light(pos=sp+vector(0,0,-0.5), color=vector(0.8,0.05,0.05))
drone_star_light = local_light(pos=sp+vector(0,0, 0.5), color=vector(0.05,0.80,0.15))
drone_belly_light= local_light(pos=sp+vector(0,-0.6,0), color=vector(0.15,0.35,1.0))

trail_c = curve(color=vector(0.05,1.0,0.40), radius=0.16)
trail_c.append(pos=sp)
trail_g = curve(color=vector(0.02,0.55,0.22), radius=0.30)
trail_g.append(pos=sp)

SONAR_R   = ALERT_DIST
sonar_ring  = ring(pos=sp, axis=vector(0,1,0), radius=SONAR_R,
                   thickness=0.12, color=vector(0.0,1.0,0.6), opacity=0.18)
sonar_spoke = cylinder(pos=sp, axis=vector(SONAR_R,0,0),
                       radius=0.09, color=vector(0.0,1.0,0.5), opacity=0.55)
sonar_angle = 0.0

def update_sonar(pos, md):
    global sonar_angle; sonar_angle += 0.07
    sx = math.cos(sonar_angle)*SONAR_R; sz = math.sin(sonar_angle)*SONAR_R
    sonar_ring.pos=pos; sonar_spoke.pos=pos; sonar_spoke.axis=vector(sx,0,sz)
    if md < OBS_BUFFER:
        sonar_ring.color=vector(1.0,0.1,0.0); sonar_ring.opacity=0.55
        sonar_spoke.color=vector(1.0,0.1,0.0)
    elif md < ALERT_DIST:
        sonar_ring.color=vector(1.0,0.85,0.0); sonar_ring.opacity=0.38
        sonar_spoke.color=vector(1.0,0.85,0.0)
    else:
        sonar_ring.color=vector(0.0,1.0,0.6); sonar_ring.opacity=0.18
        sonar_spoke.color=vector(0.0,1.0,0.5)

def move_drone(pos, hd, t_sim):
    drone_body.pos = pos
    drone_nose.pos = pos+hd*1.30; drone_nose.axis = hd*1.0
    drone_cam.pos  = pos+hd*1.38
    drone_dome.pos = pos+vector(0,0.62,0)
    drone_ll.pos   = pos+hd*1.1+vector(0,0.14, 0.35)
    drone_lr.pos   = pos+hd*1.1+vector(0,0.14,-0.35)
    drone_beam.pos = pos+hd*0.9; drone_beam.axis = hd*7.0
    drone_shad.pos = vector(pos.x, PIPE_Y+PIPE_RADIUS+0.15, pos.z)
    bp = 0.55 + 0.35*abs(math.sin(t_sim*2.5))
    drone_beacon.pos = pos+vector(0,-0.50,0); drone_beacon.opacity = bp
    for i,tf in enumerate(thr_off):
        thrusters[i].pos  = pos+tf
        props[i].pos      = pos+tf+vector(0,0.02,0)
        thr_glows[i].pos  = pos+tf+vector(0,-0.05,0)
    drone_spotlight.pos  = pos + hd*3.5
    drone_port_light.pos = pos + vector(0,0,-0.6)
    drone_star_light.pos = pos + vector(0,0, 0.6)
    drone_belly_light.pos= pos + vector(0,-0.7,0)

def spin_props(ang):
    for i, tf in enumerate(thr_off):
        props[i].rotate(angle=ang, axis=vector(0,1,0), origin=thrusters[i].pos)

def update_camera(pos, hd):
    if cam_mode == 1:
        scene.camera.pos = pos + vector(-hd.x*20, 10, -hd.z*20)
        scene.forward = norm(pos - scene.camera.pos)
    elif cam_mode == 2:
        scene.camera.pos = pos + vector(0, 50, 0)
        scene.forward = vector(0,-1,0)
        scene.up = norm(hd) if mag(hd) > 0.01 else vector(1,0,0)
    elif cam_mode == 3:
        scene.camera.pos = pos + hd*1.8 + vector(0,0.4,0)
        scene.forward = hd; scene.up = vector(0,1,0)

# ══════════════════════════════════════════════════════════════
#  SENSORS
# ══════════════════════════════════════════════════════════════
def compute_current(pos, wp_idx):
    drift=vector(0,0,0); hit_zone=-1; hit_str=0.0
    for zi,(zx,zz,zr,zs,zd) in enumerate(CURRENT_ZONES):
        d = math.sqrt((pos.x-zx)**2+(pos.z-zz)**2)
        if d < zr:
            fo = (1.0-d/zr)**1.5
            nd = norm(vector(zd.x,0,zd.z))
            drift += nd*zs*fo
            if fo*zs > hit_str: hit_str=fo*zs; hit_zone=zi
    if wp_idx >= HORIZ_WP_END:
        _,_,vr,_,vzd = VERT_CURRENT_ZONE
        d3 = math.sqrt((pos.x-VERT_PIPE_X)**2+(pos.z-VERT_PIPE_Z)**2)
        dy = abs(pos.y - (VERT_PIPE_BOT+VERT_PIPE_TOP)/2)
        if d3<vr and dy<(VERT_PIPE_TOP-VERT_PIPE_BOT)/2:
            fo=( 1.0-d3/vr)**1.5; zs_=VERT_CURRENT_ZONE[3]
            nd=norm(vector(vzd.x,vzd.y,vzd.z)); drift+=nd*zs_*fo
            if fo*zs_ > hit_str: hit_str=fo*zs_; hit_zone=len(CURRENT_ZONES)
    return drift, hit_zone, hit_str

def update_current_arrows(t_sim):
    for i,ca in enumerate(current_arrows):
        ph=(t_sim*1.4+i*0.7)%(2*math.pi)
        ca['obj'].opacity=0.12+0.22*abs(math.sin(ph))
    for i,ca in enumerate(vert_cur_arrows):
        ph=(t_sim*1.8+i*0.5)%(2*math.pi)
        ca['obj'].opacity=0.10+0.28*abs(math.sin(ph))

def compute_turbidity(t_sim,in_zone,cur_str):
    base=BASE_TURBIDITY+6.0*math.sin(t_sim*0.12)
    return round(min(max(base+(cur_str*55.0 if in_zone>=0 else 0)+random.uniform(-1.5,1.5),0),150),1)

def turbidity_label(ntu):
    if ntu<10: return "CLEAR    "
    if ntu<25: return "SLIGHT   "
    if ntu<50: return "MODERATE "
    if ntu<80: return "HIGH     "
    return             "VERY HIGH"

def compute_water_temp(t_sim,pos_y,cur_str):
    depth = max(0.0, SEA_SURFACE_Y - pos_y)
    return round(min(max(22.0 - depth*0.008 + 1.2*math.sin(t_sim*0.08)
                         + cur_str*2.5 + random.uniform(-0.15,0.15), 2), 35), 2)

def compute_pressure(pos_y):
    depth = max(0.0, SEA_SURFACE_Y - pos_y)
    return round(1.013 + depth * 0.1, 3)

def compute_dissolved_oxygen(t_sim,temp_c,cur_str):
    return round(min(max(9.5-(temp_c-15.0)*0.12+0.4*math.sin(t_sim*0.11+1.2)
                         -cur_str*1.8+random.uniform(-0.10,0.10),2),14),2)

def compute_ph(t_sim,do_val,cur_str):
    return round(min(max(8.10+0.06*math.sin(t_sim*0.07+0.5)
                         +(do_val-7.5)*0.02-cur_str*0.08+random.uniform(-0.02,0.02),7),9),3)

def compute_salinity(t_sim,pos_y,cur_str):
    depth = max(0.0, SEA_SURFACE_Y - pos_y)
    return round(min(max(34.5+depth*0.002+0.3*math.sin(t_sim*0.06+2.0)
                         +cur_str*0.5+random.uniform(-0.05,0.05),30),40),3)

def compute_angles(hd):
    hx,hy,hz = hd.x, hd.y, hd.z
    theta = math.degrees(math.atan2(hz, hx))
    alpha = math.degrees(math.atan2(hy, math.sqrt(hx*hx+hz*hz)))
    zeta  = math.degrees(math.atan2(hz, hy))
    return theta, alpha, zeta

# ══════════════════════════════════════════════════════════════
#  GRAPHS
# ══════════════════════════════════════════════════════════════
def show_angle_and_height_graph(steps, theta_log, alpha_log, zeta_log,
                                height_log, horiz_end_idx):
    BG    ='#030e1a'; PANEL='#061520'; GRID='#0c2235'; TICK='#3a7090'
    C_THETA='#00e5ff'; C_ALPHA='#76ff03'; C_ZETA='#ff9100'
    C_HEIGHT='#ff66cc'; C_TRANS='#ff1744'

    x = np.array(steps)
    MAX_PTS = 4000
    if len(x) > MAX_PTS:
        idx = np.linspace(0,len(x)-1,MAX_PTS,dtype=int)
        x          = x[idx]
        theta_arr  = np.array(theta_log)[idx]
        alpha_arr  = np.array(alpha_log)[idx]
        zeta_arr   = np.array(zeta_log)[idx]
        height_arr = np.array(height_log)[idx]
    else:
        theta_arr  = np.array(theta_log)
        alpha_arr  = np.array(alpha_log)
        zeta_arr   = np.array(zeta_log)
        height_arr = np.array(height_log)

    fig = plt.figure(figsize=(16, 22), facecolor=BG, dpi=100)
    fig.suptitle('DRONE TRAJECTORY ANALYSIS  ·  Angles + Altitude  ·  v17.0',
                 color='#00ffcc', fontsize=17, fontweight='bold',
                 fontfamily='monospace', y=0.988)
    gs = gridspec.GridSpec(5, 1, figure=fig,
                           left=0.08, right=0.97,
                           top=0.970, bottom=0.035, hspace=0.50)
    configs = [
        (gs[0], theta_arr, C_THETA,  'θ  THETA  —  Azimuth / Yaw',  'Theta θ  (°)',  '', (-185,185)),
        (gs[1], alpha_arr, C_ALPHA,  'α  ALPHA  —  Elevation / Pitch','Alpha α  (°)', '', (-95,95)),
        (gs[2], zeta_arr,  C_ZETA,   'ζ  ZETA   —  Roll / Bank',     'Zeta ζ  (°)',  '', (-185,185)),
        (gs[3], height_arr,C_HEIGHT, '📏 HEIGHT ABOVE SEA LEVEL',    'Height ASL (m)','', (None,None)),
    ]
    def annotate_ax(ax, x_arr, y_arr, col, ylabel, subtitle, ylim, title):
        ax.set_facecolor(PANEL)
        for sp_ in ax.spines.values(): sp_.set_color(GRID)
        ax.tick_params(colors=TICK, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.7, linestyle='--', alpha=0.8)
        if ylim[0] is not None: ax.set_ylim(ylim)
        ax.set_xlim(x_arr[0], x_arr[-1])
        ax.plot(x_arr, y_arr, color=col, linewidth=1.3, alpha=0.95, zorder=3)
        ax.fill_between(x_arr, y_arr, alpha=0.12, color=col, zorder=2)
        ax.axhline(0, color='#fff', linewidth=0.5, linestyle=':', alpha=0.35)
        mean_v = float(np.mean(y_arr))
        ax.axhline(mean_v, color='#fff', linewidth=1.0, linestyle='--', alpha=0.65, label=f'Mean = {mean_v:.1f}')
        mn,mx = float(np.min(y_arr)), float(np.max(y_arr))
        imin,imax = int(np.argmin(y_arr)), int(np.argmax(y_arr))
        rng = mx - mn if mx != mn else 1
        ax.annotate(f'MIN {mn:.1f}', xy=(x_arr[imin],mn), xytext=(x_arr[imin], mn-rng*0.18),
                    color='#ff4444', fontsize=7.5, fontfamily='monospace', ha='center', va='top',
                    arrowprops=dict(arrowstyle='->', color='#ff4444', lw=0.8))
        ax.annotate(f'MAX {mx:.1f}', xy=(x_arr[imax],mx), xytext=(x_arr[imax], mx+rng*0.18),
                    color='#ff4444', fontsize=7.5, fontfamily='monospace', ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->', color='#ff4444', lw=0.8))
        if horiz_end_idx>0 and horiz_end_idx<len(steps):
            tx=steps[min(horiz_end_idx, len(steps)-1)]
            ax.axvline(tx, color=C_TRANS, lw=1.5, ls='--', alpha=0.80, label='↑ Riser')
        stats=(f'  Mean:{mean_v:>9.2f}\n  Std:{float(np.std(y_arr)):>9.2f}\n  Min:{mn:>9.2f}\n  Max:{mx:>9.2f}')
        ax.text(0.005,0.97,stats, transform=ax.transAxes, fontsize=8, color='#aaddee',
                fontfamily='monospace', va='top',
                bbox=dict(boxstyle='round,pad=0.4',facecolor='#030e1a',alpha=0.82,edgecolor='#0c2235',lw=0.8))
        ax.set_title(title, color=col, fontsize=11, fontweight='bold', fontfamily='monospace', loc='left', pad=7)
        ax.set_ylabel(ylabel, color=TICK, fontsize=9, fontfamily='monospace', labelpad=7)
        ax.set_xlabel('Mission Step', color=TICK, fontsize=9, fontfamily='monospace')
        leg=ax.legend(loc='upper right', facecolor='#030e1a', edgecolor='#0c2235',
                      labelcolor='#ccddee', fontsize=8.5, framealpha=0.90)
        for t_ in leg.get_texts(): t_.set_fontfamily('monospace')

    for (gs_pos, arr, col, title, ylabel, subtitle, ylim) in configs:
        annotate_ax(fig.add_subplot(gs_pos), x, arr, col, ylabel, subtitle, ylim, title)

    ax_ov = fig.add_subplot(gs[4]); ax_ov.set_facecolor(PANEL)
    for sp_ in ax_ov.spines.values(): sp_.set_color(GRID)
    ax_ov.tick_params(colors=TICK, labelsize=9)
    ax_ov.grid(True, color=GRID, linewidth=0.7, linestyle='--', alpha=0.8)
    ax_ov.set_xlim(x[0], x[-1])
    ax_ov.plot(x, theta_arr, color=C_THETA, lw=1.3, alpha=0.90, label='θ Theta')
    ax_ov.plot(x, alpha_arr, color=C_ALPHA, lw=1.3, alpha=0.90, label='α Alpha')
    ax_ov.plot(x, zeta_arr,  color=C_ZETA,  lw=1.3, alpha=0.90, label='ζ Zeta')
    ax_ov.axhline(0, color='#fff', lw=0.5, ls=':', alpha=0.35)
    if horiz_end_idx>0 and horiz_end_idx<len(steps):
        ax_ov.axvline(steps[min(horiz_end_idx,len(steps)-1)], color=C_TRANS, lw=1.5, ls='--', alpha=0.80, label='↑ Riser')
    ax_ov.set_title('OVERLAY — All Angles', color='#ffffff', fontsize=11,
                    fontweight='bold', fontfamily='monospace', loc='left', pad=7)
    ax_ov.set_ylabel('Angle (°)', color=TICK, fontsize=9, fontfamily='monospace')
    ax_ov.set_xlabel('Mission Step', color=TICK, fontsize=9, fontfamily='monospace')
    leg_ov=ax_ov.legend(loc='upper right', facecolor='#030e1a', edgecolor='#0c2235',
                        labelcolor='#ccddee', fontsize=9, framealpha=0.92)
    for t_ in leg_ov.get_texts(): t_.set_fontfamily('monospace')
    plt.savefig('angle_height_graph.png', dpi=110, facecolor=BG, bbox_inches='tight')
    print("  Angle + Height graph saved → angle_height_graph.png")
    plt.show()


def show_pipe_temp_graph(steps, pipe_temp_log, nearest_pipe_log, horiz_end_idx):
    BG='#030e1a'; PANEL='#061520'; GRID='#0c2235'; TICK='#3a7090'; C_TRANS='#ff1744'
    def seg_col_mpl(t):
        if t>=110: return '#ff0d00'
        if t>= 90: return '#ff5900'
        if t>= 70: return '#ffb800'
        if t>= 55: return '#f0f024'
        if t>= 45: return '#66ee66'
        if t>= 35: return '#00ccff'
        return '#2244ff'

    x=np.array(steps); pt=np.array(pipe_temp_log); nd=np.array(nearest_pipe_log)
    MAX_PTS=3000
    if len(x)>MAX_PTS:
        idx=np.linspace(0,len(x)-1,MAX_PTS,dtype=int)
        x=x[idx]; pt=pt[idx]; nd=nd[idx]

    from matplotlib.collections import LineCollection
    fig=plt.figure(figsize=(16,14), facecolor=BG, dpi=100)
    fig.suptitle('🌡️  PIPELINE TEMPERATURE ANALYSIS  ·  v17.0',
                 color='#ff8844', fontsize=17, fontweight='bold', fontfamily='monospace', y=0.985)
    gs=gridspec.GridSpec(3,2,figure=fig,left=0.07,right=0.97,top=0.965,bottom=0.05,hspace=0.55,wspace=0.30)

    ax1=fig.add_subplot(gs[0,:]); ax1.set_facecolor(PANEL)
    for sp_ in ax1.spines.values(): sp_.set_color(GRID)
    ax1.tick_params(colors=TICK,labelsize=9); ax1.grid(True,color=GRID,lw=0.7,ls='--',alpha=0.8)
    pts_lc=np.array([x,pt]).T.reshape(-1,1,2); segs_lc=np.concatenate([pts_lc[:-1],pts_lc[1:]],axis=1)
    norm_lc=plt.Normalize(30,125)
    lc=LineCollection(segs_lc,cmap='plasma',norm=norm_lc,linewidth=2.2,alpha=0.95,zorder=3)
    lc.set_array(pt); ax1.add_collection(lc)
    ax1.set_xlim(x[0],x[-1]); ax1.set_ylim(20,135); ax1.fill_between(x,pt,20,alpha=0.10,color='#ff6600')
    ax1.axhline(PIPE_TEMP_ALERT_HOT,color='#ff1100',lw=1.2,ls='--',alpha=0.75,label=f'Critical HOT ({PIPE_TEMP_ALERT_HOT}°C)')
    ax1.axhline(PIPE_TEMP_WARN_HOT,color='#ff8800',lw=1.0,ls='--',alpha=0.65,label=f'Warning HOT ({PIPE_TEMP_WARN_HOT}°C)')
    ax1.axhline(PIPE_TEMP_WARN_COOL,color='#00ccff',lw=1.0,ls='--',alpha=0.65,label=f'Warning COOL ({PIPE_TEMP_WARN_COOL}°C)')
    ax1.axhline(PIPE_TEMP_ALERT_COOL,color='#4466ff',lw=1.2,ls='--',alpha=0.75,label=f'Critical COOL ({PIPE_TEMP_ALERT_COOL}°C)')
    if horiz_end_idx>0 and horiz_end_idx<len(steps):
        tx=steps[min(horiz_end_idx,len(steps)-1)]; ax1.axvline(tx,color=C_TRANS,lw=1.5,ls='--',alpha=0.80,label='↑ Riser')
    imax=int(np.argmax(pt)); imin=int(np.argmin(pt))
    ax1.annotate(f'MAX {pt[imax]:.1f}°C',xy=(x[imax],pt[imax]),xytext=(x[imax],pt[imax]+9),
                 color='#ff4444',fontsize=8,fontfamily='monospace',ha='center',
                 arrowprops=dict(arrowstyle='->',color='#ff4444',lw=0.9))
    ax1.annotate(f'MIN {pt[imin]:.1f}°C',xy=(x[imin],pt[imin]),xytext=(x[imin],pt[imin]-11),
                 color='#44aaff',fontsize=8,fontfamily='monospace',ha='center',
                 arrowprops=dict(arrowstyle='->',color='#44aaff',lw=0.9))
    cbar=fig.colorbar(lc,ax=ax1,pad=0.01,fraction=0.012)
    cbar.set_label('Pipe Temp (°C)',color=TICK,fontfamily='monospace',fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TICK,labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color=TICK,fontfamily='monospace')
    ax1.set_title('PIPE INTERNAL TEMPERATURE',color='#ff8844',fontsize=12,fontweight='bold',fontfamily='monospace',loc='left',pad=8)
    ax1.set_ylabel('Temperature (°C)',color=TICK,fontsize=10,fontfamily='monospace',labelpad=8)
    ax1.set_xlabel('Mission Step',color=TICK,fontsize=9,fontfamily='monospace')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
    leg1=ax1.legend(loc='upper right',facecolor='#030e1a',edgecolor='#331100',labelcolor='#ffccaa',fontsize=8.5,framealpha=0.90)
    for t_ in leg1.get_texts(): t_.set_fontfamily('monospace')

    ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor(PANEL)
    for sp_ in ax2.spines.values(): sp_.set_color(GRID)
    ax2.tick_params(colors=TICK,labelsize=9); ax2.grid(True,color=GRID,lw=0.7,ls='--',alpha=0.8)
    ax2.plot(x,nd,color='#00ffaa',lw=1.2,alpha=0.85); ax2.fill_between(x,nd,alpha=0.10,color='#00ffaa')
    ax2.axhline(6.5,color='#ffaa00',lw=1.0,ls='--',alpha=0.70,label='Read range (6.5 u)')
    ax2.set_title('Proximity to Nearest Pipe Segment',color='#00ffaa',fontsize=10,fontweight='bold',fontfamily='monospace',loc='left',pad=6)
    ax2.set_ylabel('Distance (m)',color=TICK,fontsize=9,fontfamily='monospace',labelpad=6)
    ax2.set_xlabel('Mission Step',color=TICK,fontsize=9,fontfamily='monospace')
    ax2.legend(facecolor='#030e1a',edgecolor='#003322',labelcolor='#aaffcc',fontsize=8,framealpha=0.85)
    for t_ in ax2.get_legend().get_texts(): t_.set_fontfamily('monospace')

    ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor(PANEL)
    for sp_ in ax3.spines.values(): sp_.set_color(GRID)
    ax3.tick_params(colors=TICK,labelsize=9); ax3.grid(True,color=GRID,lw=0.7,ls='--',alpha=0.6,axis='y')
    close_mask=nd<8.0; pt_close=pt[close_mask] if close_mask.any() else pt
    bins=np.linspace(20,135,24); counts,edges,patches_=ax3.hist(pt_close,bins=bins,edgecolor='#030e1a',lw=0.4)
    for patch,left in zip(patches_,edges[:-1]): patch.set_facecolor(seg_col_mpl(left)); patch.set_alpha(0.88)
    ax3.axvline(PIPE_TEMP_ALERT_HOT,color='#ff1100',lw=1.1,ls='--',alpha=0.75)
    ax3.axvline(PIPE_TEMP_WARN_HOT,color='#ff8800',lw=1.0,ls='--',alpha=0.65)
    ax3.axvline(PIPE_TEMP_WARN_COOL,color='#00ccff',lw=1.0,ls='--',alpha=0.65)
    ax3.axvline(PIPE_TEMP_ALERT_COOL,color='#4466ff',lw=1.1,ls='--',alpha=0.75)
    ax3.set_title('Temp Distribution (close reads)',color='#ffcc66',fontsize=10,fontweight='bold',fontfamily='monospace',loc='left',pad=6)
    ax3.set_xlabel('Temperature (°C)',color=TICK,fontsize=9,fontfamily='monospace')
    ax3.set_ylabel('Count',color=TICK,fontsize=9,fontfamily='monospace',labelpad=6)

    ax4=fig.add_subplot(gs[2,:]); ax4.set_facecolor(PANEL)
    for sp_ in ax4.spines.values(): sp_.set_color(GRID)
    ax4.tick_params(colors=TICK,labelsize=9); ax4.grid(True,color=GRID,lw=0.7,ls='--',alpha=0.7,axis='y')
    seg_labels=([f'H-{i+1}\n{PIPE_SEG_TEMPS[i]:.1f}°C' for i in range(len(PIPE_SEG_TEMPS))]+
                [f'V-{i+1}\n{VERT_SEG_TEMPS[i]:.1f}°C' for i in range(len(VERT_SEG_TEMPS))])
    seg_temps_all=PIPE_SEG_TEMPS+VERT_SEG_TEMPS
    seg_cols=[seg_col_mpl(t) for t in seg_temps_all]
    xpos=np.arange(len(seg_labels))
    bars=ax4.bar(xpos,seg_temps_all,color=seg_cols,edgecolor='#030e1a',lw=0.6,width=0.68,alpha=0.92,zorder=3)
    for bar_,t_ in zip(bars,seg_temps_all):
        ax4.text(bar_.get_x()+bar_.get_width()/2,bar_.get_height()+1.5,f'{t_:.1f}°',
                 ha='center',va='bottom',color='#fff',fontsize=9.5,fontfamily='monospace',fontweight='bold')
    ax4.axhline(PIPE_TEMP_ALERT_HOT,color='#ff1100',lw=1.2,ls='--',alpha=0.75,label=f'Critical HOT ({PIPE_TEMP_ALERT_HOT}°C)')
    ax4.axhline(PIPE_TEMP_WARN_HOT,color='#ff8800',lw=1.0,ls='--',alpha=0.65,label=f'Warning HOT ({PIPE_TEMP_WARN_HOT}°C)')
    ax4.axhline(PIPE_TEMP_WARN_COOL,color='#00ccff',lw=1.0,ls='--',alpha=0.65,label=f'Warning COOL ({PIPE_TEMP_WARN_COOL}°C)')
    ax4.axhline(PIPE_TEMP_ALERT_COOL,color='#4466ff',lw=1.2,ls='--',alpha=0.75,label=f'Critical COOL ({PIPE_TEMP_ALERT_COOL}°C)')
    ax4.axvline(len(PIPE_SEG_TEMPS)-0.5,color='#aaa',lw=1.2,ls=':',alpha=0.55)
    ax4.text(len(PIPE_SEG_TEMPS)*0.5-0.5,3,'HORIZONTAL PIPE',ha='center',va='bottom',color='#aaa',fontsize=9,fontfamily='monospace',alpha=0.70)
    ax4.text(len(PIPE_SEG_TEMPS)+len(VERT_SEG_TEMPS)*0.5-0.5,3,'VERTICAL RISER',ha='center',va='bottom',color='#aaa',fontsize=9,fontfamily='monospace',alpha=0.70)
    ax4.set_xticks(xpos); ax4.set_xticklabels(seg_labels,fontfamily='monospace',fontsize=9)
    ax4.set_ylim(0,140); ax4.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax4.set_title('PIPE SEGMENT TEMPERATURES  —  All Segments',color='#ffaa44',fontsize=11,fontweight='bold',fontfamily='monospace',loc='left',pad=8)
    ax4.set_ylabel('Internal Fluid Temp (°C)',color=TICK,fontsize=9.5,fontfamily='monospace',labelpad=8)
    leg4=ax4.legend(loc='upper right',facecolor='#030e1a',edgecolor='#331100',labelcolor='#ffccaa',fontsize=8.5,framealpha=0.90)
    for t_ in leg4.get_texts(): t_.set_fontfamily('monospace')
    plt.savefig('pipe_temp_graph.png',dpi=110,facecolor=BG,bbox_inches='tight')
    print("  Pipeline temp graph saved → pipe_temp_graph.png")
    plt.show()

# ══════════════════════════════════════════════════════════════
#  HUD LABELS
# ══════════════════════════════════════════════════════════════
OCN_TEXT    = vector(0.60, 0.95, 1.00)
OCN_BG      = vector(0.01, 0.10, 0.22)
OCN_ALERT   = vector(1.00, 0.82, 0.08)
OCN_ALERT_BG= vector(0.12, 0.04, 0.00)

HUD_POS    = vector(-200, PIPE_Y+35, -100)
ALERT_POS  = vector(  20, PIPE_Y+90, -100)
STATUS_POS = vector(   0, PIPE_Y-15,   50)
BAT_POS    = vector( 200, PIPE_Y+90, -100)

hud_lbl      = label(pos=HUD_POS, text="INITIALISING...",
                     color=OCN_TEXT, background=OCN_BG,
                     border=10, font='monospace', height=13, box=True, line=False, opacity=0.94)
alert_lbl    = label(pos=ALERT_POS, text="",
                     color=OCN_ALERT, background=OCN_ALERT_BG,
                     border=10, font='monospace', height=14, box=True, line=False, opacity=0)
status_lbl   = label(pos=STATUS_POS, text="🌊 READY TO LAUNCH",
                     color=vector(0.35,1.0,0.70), background=vector(0.01,0.09,0.05),
                     border=7, font='monospace', height=13, box=True, line=False)
bat_lbl      = label(pos=BAT_POS, text="🔋 100%",
                     color=vector(0.35,1.0,0.70), background=OCN_BG,
                     border=8, font='monospace', height=13, box=True, line=False, opacity=0.94)
pipe_temp_lbl= label(pos=vector(200,PIPE_Y+70,-100), text="🌡️ PIPE TEMP\n  ---  °C",
                     color=vector(1.0,0.70,0.10), background=vector(0.10,0.03,0.00),
                     border=8, font='monospace', height=13, box=True, line=False, opacity=0.94)
height_lbl   = label(pos=vector(200,PIPE_Y+50,-100), text="📏 HEIGHT\n--- m ASL",
                     color=vector(0.55,0.88,1.0), background=vector(0.02,0.06,0.18),
                     border=8, font='monospace', height=13, box=True, line=False, opacity=0.94)
collision_lbl= label(pos=vector(20,PIPE_Y+75,-100), text="",
                     color=vector(1.0,0.05,0.0), background=vector(0.25,0.0,0.0),
                     border=14, font='monospace', height=18, box=True, line=False, opacity=0)

# ══════════════════════════════════════════════════════════════
#  MAIN MISSION
# ══════════════════════════════════════════════════════════════
def run_mission():
    global battery,battery_low,paused,cam_mode,turbidity,speed_mult
    global replay_requested,reset_requested,sonar_angle
    global current_event_log,total_current_hits

    battery=100.0; battery_low=False; paused=False; sonar_angle=0.0
    turbidity=BASE_TURBIDITY; current_event_log=[]; total_current_hits=0
    replay_requested=False; reset_requested=False

    pause_btn.text='  ⏸  PAUSE  '
    pause_btn.background=vector(0.0,0.18,0.30); pause_btn.color=vector(0.0,0.90,1.00)

    for i,m in enumerate(corrosion_markers):   m.color=corrosion_colors[i]; m.emissive=True; m.opacity=0.92
    for i,m in enumerate(vert_corr_markers):   m.color=vert_corr_colors[i]; m.emissive=True; m.opacity=0.92
    for i,m in enumerate(crack_markers):       m.color=crack_col_map[crack_severities[i]]; m.emissive=True; m.opacity=0.90
    for s in crack_scars:                      s.color=vector(0.05,0.01,0.01)
    for i,m in enumerate(vert_crack_markers):  m.color=crack_col_map[vert_crack_severities[i]]; m.emissive=True; m.opacity=0.90
    for s in vert_crack_scars:                 s.color=vector(0.05,0.01,0.01)

    trail_c.clear(); trail_g.clear()
    trail_c.append(pos=sp); trail_g.append(pos=sp)
    collision_lbl.opacity=0; alert_lbl.opacity=0
    sonar_ring.opacity=0.18; sonar_spoke.opacity=0.55

    for tr in all_temp_rings:
        tr['scanned']=False
        tr['inner'].opacity=0.95; tr['glow'].opacity=0.18
        tr['inner'].color=tr['base_col']; tr['glow'].color=tr['base_col']
        tr['lbl'].color=tr['base_col']
    pipe_temp_lbl.text="🌡️ PIPE TEMP\n  ---  °C"
    pipe_temp_lbl.color=vector(1.0,0.70,0.10)
    height_lbl.text="📏 HEIGHT\n--- m ASL"

    collision_timer=0; alert_timer=0
    inspected_corr=set(); inspected_crack=set()
    inspected_vcorr=set(); inspected_vcrack=set()
    total_dist=0.0; step_count=0; max_speed_ms=0.0; min_dist_ever=999.0
    corr_log=[]; crack_log=[]; vcorr_log=[]; vcrack_log=[]
    # ── Fish detection state ──────────────────────────────────
    FISH_DETECT_DIST    = 12.0   # detection radius (m)
    fish_detected_schools = set()   # school indices ever detected
    fish_detect_total   = 0         # total detection events this mission
    fish_nearest_dist   = 999.0
    fish_nearest_name   = "---"
    fish_detect_flash   = 0         # flash timer for HUD
    # Reset fish colours to normal
    for _si, _sc in enumerate(fish_schools):
        for _f in _sc["fish"]:
            _f["body"].color = _sc["body_col"]
            _f["tail"].color = _sc["fin_col"]
            _f["dors"].color = _sc["fin_col"]
    mission_start=time.time(); t_sim=0.0; prev_in_zone=-1

    angle_steps=[]; theta_log=[]; alpha_log=[]; zeta_log=[]; height_log=[]
    s_temp=22.0; s_pres=1.013; s_do=9.5; s_ph=8.10; s_sal=34.5

    pipe_temp_read_log=[]; nearest_pipe_dist_log=[]; pipe_temp_steps=[]
    current_pipe_temp=None; scanned_temp_segs=set()

    curr_pos=vector(sp.x,sp.y,sp.z); curr_head=vector(1,0,0)
    wp_idx=0; move_drone(curr_pos,curr_head,0); on_vertical=False

    status_lbl.text="🌊 READY TO LAUNCH"; status_lbl.color=vector(0.35,1.0,0.70)
    hud_lbl.text="INITIALISING...";       hud_lbl.color=OCN_TEXT
    bat_lbl.text="🔋 100%";               bat_lbl.color=vector(0.35,1.0,0.70)

    while wp_idx < len(waypoints):
        rate(1.0/DT); t_sim += DT
        update_bubbles(t_sim)
        update_seaweed(t_sim)
        update_current_arrows(t_sim)
        update_fish(t_sim)         # 🐟 NEW
        update_jellyfish(t_sim)    # 🪼 NEW
        update_plankton(t_sim)     # ✨ NEW
        on_vertical = (wp_idx >= HORIZ_WP_END)
        prop_spd = 0.08 if paused else (0.50 * speed_mult)
        spin_props(prop_spd)

        if replay_requested or reset_requested: return
        if paused:
            status_lbl.text="⏸  PAUSED — press SPACE to resume"
            status_lbl.color=vector(1.0,0.55,0.10); continue

        target=waypoints[wp_idx]; diff=target-curr_pos
        STEP = BASE_STEP * speed_mult
        if mag(diff) < STEP:
            wp_idx += 1
            if wp_idx >= len(waypoints): break
            continue

        move_dir = norm(diff)
        drift,hit_zone,hit_str = compute_current(curr_pos,wp_idx)
        in_current = (hit_zone >= 0)
        if in_current:
            cur_factor = max(0.60, 1.0-hit_str*1.8); eff_step=STEP*cur_factor
            curr_pos += drift*DT*3.5
            if hit_zone != prev_in_zone:
                total_current_hits += 1
                current_event_log.append((step_count,hit_zone))
                prev_in_zone = hit_zone
        else:
            cur_factor=1.0; eff_step=STEP
            if prev_in_zone >= 0: prev_in_zone=-1

        turbidity = compute_turbidity(t_sim,hit_zone,hit_str)
        s_temp    = compute_water_temp(t_sim,curr_pos.y,hit_str)
        s_pres    = compute_pressure(curr_pos.y)
        s_do      = compute_dissolved_oxygen(t_sim,s_temp,hit_str)
        s_ph      = compute_ph(t_sim,s_do,hit_str)
        s_sal     = compute_salinity(t_sim,curr_pos.y,hit_str)

        min_dist=999.0; lateral_push=vector(0,0,0); avoiding=False
        for opos in all_obstacle_positions:
            d = mag(curr_pos - opos)
            if d < min_dist: min_dist=d
            if d < OBS_BUFFER:
                avoiding=True
                to_xz = vector(curr_pos.x-opos.x, 0, curr_pos.z-opos.z)
                if mag(to_xz)<0.01: to_xz=vector(0,0,1)
                ff = (norm(vector(move_dir.x,0,move_dir.z))
                      if mag(vector(move_dir.x,0,move_dir.z))>0.01 else vector(1,0,0))
                side = to_xz - dot(to_xz,ff)*ff
                if mag(side)<0.01: side=vector(-ff.z,0,ff.x)
                lateral_push += norm(side)*SIDESTEP_W*((OBS_BUFFER-d)/OBS_BUFFER)**1.6

        if min_dist < min_dist_ever: min_dist_ever=min_dist
        if mag(lateral_push) > 0.001:
            bl = min(mag(lateral_push)/(SIDESTEP_W*1.8), 0.68)
            move_dir = norm(move_dir*(1.0-bl) + norm(lateral_push)*bl)
        if mag(move_dir)<0.01: move_dir=curr_head
        move_dir = norm(move_dir); curr_head=move_dir

        curr_pos  += move_dir*eff_step
        total_dist+= eff_step; step_count+=1
        spd_ms    = (eff_step/DT)*UNIT_TO_M
        if spd_ms > max_speed_ms: max_speed_ms=spd_ms
        elapsed_now = time.time()-mission_start
        avg_ms = (total_dist/elapsed_now)*UNIT_TO_M if elapsed_now>0 else 0

        move_drone(curr_pos, move_dir, t_sim)
        trail_c.append(pos=curr_pos); trail_g.append(pos=curr_pos)
        update_sonar(curr_pos, min_dist)

        height_asl = curr_pos.y - SEA_SURFACE_Y
        height_log.append(height_asl)
        angle_steps.append(step_count)
        hc = (vector(0.2,1.0,0.55) if height_asl > 0 else vector(1.0,0.60,0.10))
        height_lbl.text = (f"📏 HEIGHT\n  {height_asl:>7.1f} m ASL\n"
                           f"  {'ABOVE' if height_asl>=0 else 'BELOW'} SEA SURFACE")
        height_lbl.color = hc

        theta,alpha,zeta = compute_angles(move_dir)
        theta_log.append(theta); alpha_log.append(alpha); zeta_log.append(zeta)

        TEMP_READ_DIST = 7.0
        best_td=999.0; best_tr_idx=-1
        for ti,tr in enumerate(all_temp_rings):
            d_tr = mag(curr_pos - tr['mid'])
            if d_tr < best_td: best_td=d_tr; best_tr_idx=ti

        nearest_pipe_dist_log.append(best_td)

        if best_tr_idx>=0 and best_td<TEMP_READ_DIST:
            tr = all_temp_rings[best_tr_idx]
            current_pipe_temp = tr['temp']
            pipe_temp_read_log.append(current_pipe_temp)
            pipe_temp_steps.append(step_count)
            pulse = 0.22+0.20*abs(math.sin(t_sim*4.5))
            tr['glow'].opacity = pulse
            if best_tr_idx not in scanned_temp_segs:
                scanned_temp_segs.add(best_tr_idx); tr['scanned']=True
                tr['inner'].opacity=1.0
            lbl_ = pipe_temp_label(current_pipe_temp)
            tc_  = pipe_temp_hud_color(current_pipe_temp)
            seg_name = (f"V-{best_tr_idx-len(pipe_temp_rings)+1}"
                        if best_tr_idx>=len(pipe_temp_rings) else f"H-{best_tr_idx+1}")
            pipe_temp_lbl.text=(f"🌡️  PIPE TEMP\n  {current_pipe_temp:.1f} °C\n"
                                f"  {lbl_}\n  Seg {seg_name} | d={best_td:.1f}m")
            pipe_temp_lbl.color=tc_
            pipe_temp_lbl.background=(
                vector(0.20,0.02,0.00) if current_pipe_temp>=PIPE_TEMP_WARN_HOT
                else vector(0.00,0.04,0.18) if current_pipe_temp<=PIPE_TEMP_WARN_COOL
                else vector(0.05,0.08,0.02))
            if current_pipe_temp>=PIPE_TEMP_ALERT_HOT:
                alert_lbl.text=f"🔥 CRITICAL HOT PIPE — Seg {seg_name}: {current_pipe_temp:.1f}°C"
                alert_lbl.color=vector(1.0,0.05,0.00); alert_lbl.opacity=1; alert_timer=80
            elif current_pipe_temp>=PIPE_TEMP_WARN_HOT and alert_timer==0:
                alert_lbl.text=f"⚠  HOT PIPE — Seg {seg_name}: {current_pipe_temp:.1f}°C"
                alert_lbl.color=vector(1.0,0.55,0.10); alert_lbl.opacity=1; alert_timer=50
            elif current_pipe_temp<=PIPE_TEMP_ALERT_COOL:
                alert_lbl.text=f"❄️  CRITICAL COOL PIPE — Seg {seg_name}: {current_pipe_temp:.1f}°C"
                alert_lbl.color=vector(0.30,0.60,1.00); alert_lbl.opacity=1; alert_timer=80
        else:
            if current_pipe_temp is not None:
                pipe_temp_lbl.text=(f"🌡️  PIPE TEMP\n  {current_pipe_temp:.1f} °C  (last)\n"
                                    f"  {pipe_temp_label(current_pipe_temp)}\n  d={best_td:.1f}m  (out of range)")
                pipe_temp_lbl.color=vector(0.55,0.55,0.55)
            pipe_temp_read_log.append(current_pipe_temp if current_pipe_temp is not None else float('nan'))
            pipe_temp_steps.append(step_count)
            for ti,tr in enumerate(all_temp_rings):
                if ti!=best_tr_idx: tr['glow'].opacity=max(tr['glow'].opacity-0.02,0.18)

        battery = max(0.0, battery - BATTERY_DRAIN*speed_mult)
        bc = (vector(0.2,1.0,0.55) if battery>50
              else vector(1.0,0.85,0.10) if battery>20
              else vector(1.0,0.15,0.05))
        bat_lbl.text=f"{'🔋' if battery>20 else '⚠'} {battery:.1f}%"; bat_lbl.color=bc

        for ci,cs in enumerate(corrosion_spots):
            if ci not in inspected_corr and mag(curr_pos-cs)<7.0:
                inspected_corr.add(ci); corrosion_markers[ci].color=vector(0.08,1.0,0.30)
                corr_log.append((step_count,ci,corrosion_severity[ci]))
                alert_lbl.text=f" CORROSION {ci+1} [{corrosion_severity[ci].upper()}]"
                alert_lbl.color=vector(0.15,1.0,0.50); alert_lbl.opacity=1; alert_timer=90
        for ci,cs in enumerate(vert_corr_spots):
            if ci not in inspected_vcorr and mag(curr_pos-cs)<6.5:
                inspected_vcorr.add(ci); vert_corr_markers[ci].color=vector(0.08,1.0,0.30)
                vcorr_log.append((step_count,ci,vert_corr_severity[ci]))
                alert_lbl.text=f" RISER CORROSION {ci+1} [{vert_corr_severity[ci].upper()}]"
                alert_lbl.color=vector(0.15,1.0,0.50); alert_lbl.opacity=1; alert_timer=90
        for ci,cp in enumerate(crack_positions):
            if ci not in inspected_crack and mag(curr_pos-cp)<7.0:
                inspected_crack.add(ci); crack_markers[ci].color=CRACK_SCANNED_COL
                crack_scars[ci].color=vector(0.60,0.04,0.01)
                crack_log.append((step_count,ci,crack_severities[ci]))
                alert_lbl.text=f" ⚠ CRACK {ci+1} [{crack_severities[ci].upper()}]"
                alert_lbl.color=vector(1.0,0.40,0.10); alert_lbl.opacity=1; alert_timer=110
        for ci,cp in enumerate(vert_crack_positions):
            if ci not in inspected_vcrack and mag(curr_pos-cp)<6.5:
                inspected_vcrack.add(ci); vert_crack_markers[ci].color=CRACK_SCANNED_COL
                vert_crack_scars[ci].color=vector(0.60,0.04,0.01)
                vcrack_log.append((step_count,ci,vert_crack_severities[ci]))
                alert_lbl.text=f" ⚠ RISER CRACK {ci+1} [{vert_crack_severities[ci].upper()}]"
                alert_lbl.color=vector(1.0,0.30,0.05); alert_lbl.opacity=1; alert_timer=110

        # ── 🐟 FISH DETECTION ────────────────────────────────────
        fish_nearest_dist = 999.0
        fish_nearest_name = "---"
        currently_detecting = False
        for _si, _sc in enumerate(fish_schools):
            sc_pos = vector(_sc["cx"], _sc["cy"], _sc["cz"])
            d_fish = mag(curr_pos - sc_pos)
            if d_fish < fish_nearest_dist:
                fish_nearest_dist = d_fish
                fish_school_colors = ["ORANGE","BLUE","PURPLE","GREEN","RED"]
                fish_nearest_name = fish_school_colors[_si % len(fish_school_colors)]
            if d_fish < FISH_DETECT_DIST:
                currently_detecting = True
                # Flash fish bodies white/yellow on detection
                flash_col = vector(1.0, 1.0, 0.3) if int(t_sim*8)%2==0 else _sc["body_col"]
                for _f in _sc["fish"]:
                    _f["body"].color = flash_col
                if _si not in fish_detected_schools:
                    fish_detected_schools.add(_si)
                    fish_detect_total += 1
                    alert_lbl.text = (f"🐟 FISH DETECTED — {fish_nearest_name} school"
                                      f" | dist={d_fish:.1f}m")
                    alert_lbl.color  = vector(0.20, 1.00, 0.55)
                    alert_lbl.opacity = 1; alert_timer = 70
                    fish_detect_flash = 50
            else:
                # Restore normal colour when drone moves away
                for _f in _sc["fish"]:
                    _f["body"].color = _sc["body_col"]

        if fish_detect_flash > 0:
            fish_detect_flash -= 1

        if avoiding and min_dist<OBS_BUFFER:
            collision_lbl.text="COLLISION WARNING — SIDESTEP ACTIVE"
            collision_lbl.opacity=0.96; collision_timer=35
        elif collision_timer>0:
            collision_timer-=1
            if collision_timer==0: collision_lbl.opacity=0

        if avoiding and min_dist<OBS_BUFFER:
            alert_lbl.text="⚠  OBSTACLE — EVADING"
            alert_lbl.color=vector(1.0,0.18,0.04); alert_lbl.opacity=1; alert_timer=28
            status_lbl.text="⚠  DRONE: AVOIDING OBSTACLE"; status_lbl.color=vector(1.0,0.30,0.06)
        elif in_current:
            zn = 'VERT' if hit_zone==len(CURRENT_ZONES) else f"ZONE {hit_zone+1}"
            status_lbl.text=f"CURRENT {zn} — {spd_ms:.2f} m/s ({int(cur_factor*100)}%)"
            status_lbl.color=vector(0.30,0.70,1.0)
        elif min_dist<ALERT_DIST:
            if alert_timer==0:
                alert_lbl.text="  CAUTION — Obstacle nearby  "; alert_lbl.color=vector(1.0,0.85,0.10)
                alert_lbl.opacity=1; alert_timer=20
            status_lbl.text="CAUTION — Obstacle approaching"; status_lbl.color=vector(1.0,0.85,0.10)
        else:
            if alert_timer>0: alert_timer-=1
            else: alert_lbl.opacity=0
            status_lbl.text=(" 🌊 HELIX ORBIT — VERTICAL RISER" if on_vertical
                             else " 🌊 HELIX ORBIT — HORIZONTAL PIPE")
            status_lbl.color=(vector(0.40,0.95,1.0) if on_vertical else vector(0.35,1.0,0.70))

        if battery<=20 and not battery_low: battery_low=True
        if battery_low:
            status_lbl.text="⚠  LOW BATTERY — RETURN TO BASE"; status_lbl.color=vector(1.0,0.55,0.10)

        cov_c  = int(100*len(inspected_corr)/max(len(corrosion_spots),1))
        cov_k  = int(100*len(inspected_crack)/max(len(crack_positions),1))
        cov_vc = int(100*len(inspected_vcorr)/max(len(vert_corr_spots),1))
        cov_vk = int(100*len(inspected_vcrack)/max(len(vert_crack_positions),1))
        tl = turbidity_label(turbidity)
        tc = (vector(0.40,1.00,0.75) if turbidity<25
              else vector(1.00,0.85,0.10) if turbidity<50
              else vector(1.00,0.30,0.05))
        zn_ = 'VERT' if hit_zone==len(CURRENT_ZONES) else (f'ZONE{hit_zone+1}' if in_current else 'CLEAR')
        sl_ = 'VERTICAL RISER' if on_vertical else 'HORIZONTAL PIPE'

        hud_lbl.text=(
            f'┌── HUD+ENV ─ CAM[{cam_mode}] ───────────────────┐\n'
            f'  SECTION   : {sl_:<18s}      \n'
            f'  SPEED     : {spd_ms:>7.2f} m/s  ×{speed_mult:.1f}      \n'
            f'  AVG SPEED : {avg_ms:>7.2f} m/s              \n'
            f'  DISTANCE  : {total_dist*UNIT_TO_M:>7.1f} m                \n'
            f'  POS       : ({curr_pos.x:>6.1f},{curr_pos.y:>6.1f},{curr_pos.z:>4.1f}) \n'
            f'  HEIGHT ASL: {height_asl:>7.1f} m               \n'
            f'  WP        : {wp_idx:>4d}/{len(waypoints):<4d}                \n'
            f'  ── ANGLES ──────────────────────────────\n'
            f'  θ THETA   : {theta:>8.2f}°  Yaw        \n'
            f'  α ALPHA   : {alpha:>8.2f}°  Pitch      \n'
            f'  ζ ZETA    : {zeta:>8.2f}°  Roll       \n'
            f'  ── INSPECTION ──────────────────────────\n'
            f'  H-CORR    : {len(inspected_corr):>2d}/{len(corrosion_spots):<2d} ({cov_c:>3d}%)       \n'
            f'  V-CORR    : {len(inspected_vcorr):>2d}/{len(vert_corr_spots):<2d} ({cov_vc:>3d}%)       \n'
            f'  H-CRACK   : {len(inspected_crack):>2d}/{len(crack_positions):<2d} ({cov_k:>3d}%)       \n'
            f'  V-CRACK   : {len(inspected_vcrack):>2d}/{len(vert_crack_positions):<2d} ({cov_vk:>3d}%)       \n'
            f'  ── PIPE TEMP ───────────────────────────\n'
            f'  PIPE °C   : {f"{current_pipe_temp:.1f} °C" if current_pipe_temp else "  --- scanning":<16s}\n'
            f'  STATUS    : {pipe_temp_label(current_pipe_temp) if current_pipe_temp else " OUT OF RANGE  ":<16s}\n'
            f'  SCANNED   : {len(scanned_temp_segs):>2d}/{len(all_temp_rings):<2d} segs            \n'
            f'  ── 🐟 FISH DETECTION SENSOR ────────────\n'
            f'  SCHOOLS   : {len(fish_detected_schools):>2d}/{len(fish_schools):<2d} detected         \n'
            f'  NEAREST   : {fish_nearest_dist:>6.1f} m  ({fish_nearest_name:<8s})  \n'
            f'  RANGE     : {FISH_DETECT_DIST:.0f} m  {"🔴 FISH IN RANGE!" if fish_nearest_dist<FISH_DETECT_DIST else "  clear         "}\n'
            f'  TOTAL HITS: {fish_detect_total:>3d} detections         \n'
            f'  🪼 JELLY  : {len(jellyfish_list)} jellyfish drifting    \n'
            f'  ✨ PLANKT : {len(plankton)} biolum. motes       \n'
            f'  ── ENV ─────────────────────────────────\n'
            f'  TURBIDITY : {turbidity:>6.1f} NTU {tl}  \n'
            f'  TEMP      : {s_temp:>6.2f} °C             \n'
            f'  PRESSURE  : {s_pres:>6.2f} bar            \n'
            f'  DISS O2   : {s_do:>6.2f} mg/L           \n'
            f'  pH        : {s_ph:>6.3f}               \n'
            f'  SALINITY  : {s_sal:>6.3f} PSU           \n'
            f'  CURRENT   : {zn_:<8s} str={hit_str:.3f}  \n'
            f'  ── SAFETY ──────────────────────────────\n'
            f'  NEAREST   : {min_dist_ever:>6.1f} m obstacle    \n'
            f'  BATTERY   : {battery:>5.1f}%                \n'
            f'└────────────────────────────────────────────┘'
        )
        hud_lbl.color = tc
        if cam_mode != 1: update_camera(curr_pos, move_dir)

    if replay_requested or reset_requested: return

    elapsed = time.time() - mission_start
    avg_ms_f = (total_dist/elapsed)*UNIT_TO_M if elapsed>0 else 0
    alert_lbl.opacity=0; collision_lbl.opacity=0
    sonar_ring.opacity=0; sonar_spoke.opacity=0

    th_a = theta_log  if theta_log  else [0]
    al_a = alpha_log  if alpha_log  else [0]
    ze_a = zeta_log   if zeta_log   else [0]
    ht_a = height_log if height_log else [0]
    cov_c  = int(100*len(inspected_corr)/max(len(corrosion_spots),1))
    cov_k  = int(100*len(inspected_crack)/max(len(crack_positions),1))
    cov_vc = int(100*len(inspected_vcorr)/max(len(vert_corr_spots),1))
    cov_vk = int(100*len(inspected_vcrack)/max(len(vert_crack_positions),1))

    hud_lbl.text=(
        f"┌──── FINAL REPORT ─────────────────────────────┐\n"
        f"  ── FLIGHT ───────────────────────────────\n"
        f"  Distance    : {total_dist*UNIT_TO_M:>8.1f} m\n"
        f"  Avg Speed   : {avg_ms_f:>8.2f} m/s\n"
        f"  Max Speed   : {max_speed_ms:>8.2f} m/s\n"
        f"  ── HEIGHT ASL ───────────────────────────\n"
        f"  Min Height  : {min(ht_a):>8.1f} m\n"
        f"  Max Height  : {max(ht_a):>8.1f} m\n"
        f"  Mean Height : {sum(ht_a)/len(ht_a):>8.1f} m\n"
        f"  ── ANGLES SUMMARY ───────────────────────\n"
        f"  θ Theta : {min(th_a):>7.1f}° to {max(th_a):>6.1f}°\n"
        f"  α Alpha : {min(al_a):>7.1f}° to {max(al_a):>6.1f}°\n"
        f"  ζ Zeta  : {min(ze_a):>7.1f}° to {max(ze_a):>6.1f}°\n"
        f"  ── PIPE TEMPERATURE ─────────────────────\n"
        f"  Segs scanned: {len(scanned_temp_segs)}/{len(all_temp_rings)}\n"
        f"  Hottest seg : {max(PIPE_SEG_TEMPS+VERT_SEG_TEMPS):.1f} °C\n"
        f"  Coolest seg : {min(PIPE_SEG_TEMPS+VERT_SEG_TEMPS):.1f} °C\n"
        f"  ── INSPECTION ───────────────────────────\n"
        f"  H-Corr  : {len(inspected_corr)}/{len(corrosion_spots)} ({cov_c}%)\n"
        f"  H-Crack : {len(inspected_crack)}/{len(crack_positions)} ({cov_k}%)\n"
        f"  V-Corr  : {len(inspected_vcorr)}/{len(vert_corr_spots)} ({cov_vc}%)\n"
        f"  V-Crack : {len(inspected_vcrack)}/{len(vert_crack_positions)} ({cov_vk}%)\n"
        f"  ── SAFETY ───────────────────────────────\n"
        f"  Closest Obs : {min_dist_ever:>5.1f} m\n"
        f"  Battery Left: {battery:>5.1f}%\n"
        f"  📊 Graph windows opening...              \n"
        f"└──────────────────────────────────────────────┘"
    )
    hud_lbl.color=vector(0.40,1.0,0.75); hud_lbl.background=vector(0.01,0.10,0.06)
    status_lbl.text="✅ MISSION COMPLETE — Graph windows opening!"
    status_lbl.color=vector(0.35,1.0,0.70)
    bat_lbl.text=f"🔋 {battery:.1f}% remaining"; bat_lbl.color=vector(0.35,1.0,0.70)

    print(f"\n{'='*60}")
    print("  MISSION COMPLETE — v17.0  (Marine Life Edition)")
    print(f"  Fish schools active  : {len(fish_schools)} schools × 8 fish")
    print(f"  Jellyfish active     : {len(jellyfish_list)}")
    print(f"  Biolum. plankton     : {len(plankton)} motes")
    print(f"  Height ASL: {min(ht_a):.1f} m → {max(ht_a):.1f} m")
    print(f"{'='*60}")

    threading.Thread(
        target=show_angle_and_height_graph,
        args=(angle_steps, theta_log, alpha_log, zeta_log, height_log, HORIZ_WP_END),
        daemon=True
    ).start()

    clean_steps=[]; clean_temps=[]; clean_dists=[]
    for s_,t_,d_ in zip(pipe_temp_steps, pipe_temp_read_log, nearest_pipe_dist_log):
        if t_ is not None and not (isinstance(t_,float) and math.isnan(t_)):
            clean_steps.append(s_); clean_temps.append(t_); clean_dists.append(d_)
    if len(clean_steps)>5:
        threading.Thread(
            target=show_pipe_temp_graph,
            args=(clean_steps, clean_temps, clean_dists, HORIZ_WP_END),
            daemon=True
        ).start()

    while True:
        rate(60)
        update_bubbles(0)
        update_seaweed(0)
        update_fish(0.0)
        update_jellyfish(0.0)
        update_plankton(0.0)
        spin_props(0.08)
        if replay_requested or reset_requested: return


while True:
    replay_requested=False; reset_requested=False
    run_mission()
    if reset_requested:
        scene.camera.pos=vector(0, 80, 120)
        scene.forward=vector(0,-0.35,-1)
        scene.up=vector(0,1,0); cam_mode=1
