import pygame
import sys
import os
import threading
import json
import time
import math
import random
import urllib.request
from collections import deque

# ── Constants ─────────────────────────────────────────────────────────────────
GAME_W   = 1024          # playable area width
CHAT_W   = 280           # AI chat panel width
SCREEN_W = GAME_W + CHAT_W   # total window width
SCREEN_H = 600
FPS          = 60
GRAVITY      = 0.55
JUMP_FORCE   = -14
VLAD_JUMP    = -22
PLAYER_SPEED = 4
VLAD_SPEED   = 5
OLLAMA_MODEL  = "llama3.1"   # change to "gemma3", "qwen2", "mistral" etc.
VISION_RANGE  = 8 * 48
AI_INTERVAL   = 3.0     # seconds between Ollama calls (normal)
AI_PANIC      = 1.0     # seconds between Ollama calls when player is close
PANIC_RANGE   = 4 * 48  # pixel distance that triggers panic replan
ROAM_INTERVAL = 6.0     # seconds between random roam destinations
TILE         = 48
JUMP_TILES   = 4        # max tiles Vlad can jump up in nav graph

WHITE = (255, 255, 255)
GOLD  = (218, 165,  32)
RED   = (180,  30,  30)
BROWN = ( 90,  55,  20)

# ── Level ─────────────────────────────────────────────────────────────────────
def _load_level(path):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            lines.append(line)
    return lines

_BASE_DIR = os.path.dirname(__file__) or "."

def _load_level_index():
    """Load levels.json and return list of {file, name} dicts."""
    index_path = os.path.join(_BASE_DIR, "levels.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            data = json.load(f)
        entries = data.get("levels", [])
        result  = []
        for e in entries:
            if isinstance(e, str):
                result.append({"file": os.path.join(_BASE_DIR, e), "name": e})
            else:
                result.append({
                    "file": os.path.join(_BASE_DIR, e["file"]),
                    "name": e.get("name", e["file"]),
                })
        if result:
            return result
    # Fallback: default level
    return [{"file": os.path.join(_BASE_DIR, "level.txt"), "name": "Level 1"}]

# Globals updated by switch_level()
LEVEL         = []
LEVEL_COLS    = 0
LEVEL_ROWS    = 0
LEVEL_PIXEL_W = 0
LEVEL_PIXEL_H = 0
NAV_GRAPH     = {}
CURRENT_LEVEL_PATH = ""

def switch_level(path):
    """Load a new level file and rebuild all derived globals."""
    global LEVEL, LEVEL_COLS, LEVEL_ROWS, LEVEL_PIXEL_W, LEVEL_PIXEL_H
    global NAV_GRAPH, CURRENT_LEVEL_PATH
    CURRENT_LEVEL_PATH = path
    LEVEL         = _load_level(path)
    LEVEL_COLS    = max(len(r) for r in LEVEL)
    LEVEL_ROWS    = len(LEVEL)
    LEVEL_PIXEL_W = LEVEL_COLS * TILE
    LEVEL_PIXEL_H = LEVEL_ROWS * TILE
    NAV_GRAPH     = _build_nav_graph()

def _tile_at(col, row):
    if 0 <= row < LEVEL_ROWS and 0 <= col < len(LEVEL[row]):
        return LEVEL[row][col]
    return 'W'

def _is_solid(col, row):
    return _tile_at(col, row) in ('W', 'F', 'G')

# ── Nav graph (pre-computed once per level) ────────────────────────────────────
def _build_nav_graph():
    """
    Nodes  = (col, row) tiles where a character can stand
             (tile is open air, tile below is solid)
    Edges  = walk left/right, fall off edges, jump up platforms
    """
    # Collect all standable tiles
    ground = set()
    for row in range(LEVEL_ROWS - 1):
        for col in range(LEVEL_COLS):
            if not _is_solid(col, row) and _is_solid(col, row + 1):
                ground.add((col, row))

    graph = {n: set() for n in ground}

    for (col, row) in ground:
        # ── Walk left / right ────────────────────────────────────────────────
        for dc in (-1, 1):
            nc = col + dc
            if (nc, row) in ground:
                graph[(col, row)].add((nc, row))
            elif not _is_solid(nc, row):
                # Walk off edge → fall to lower platform
                for dr in range(1, LEVEL_ROWS - row):
                    if (nc, row + dr) in ground:
                        graph[(col, row)].add((nc, row + dr))
                        break
                    if _is_solid(nc, row + dr):
                        break

        # ── Jump straight up then land left/right/same ───────────────────────
        for jump_h in range(1, JUMP_TILES + 1):
            tr = row - jump_h
            if tr < 0:
                break
            # Column must be clear from tr up to row
            if any(_is_solid(col, r) for r in range(tr, row)):
                break
            for dc in (-1, 0, 1):
                target = (col + dc, tr)
                if target in ground:
                    graph[(col, row)].add(target)

    # Convert sets to lists
    return {n: list(neighbors) for n, neighbors in graph.items()}


def bfs_path(start, goal):
    """Return list of (col,row) tiles from start to goal, or [] if unreachable."""
    if start == goal:
        return [start]
    if start not in NAV_GRAPH or goal not in NAV_GRAPH:
        return []
    visited = {start}
    queue   = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        for nb in NAV_GRAPH.get(node, []):
            if nb == goal:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return []


def best_escape_tile(vlad_tile, player_tile, strategy):
    """
    Find the best escape tile reachable from vlad_tile given a strategy.
    strategy: 'FAR' | 'HIGH' | 'CORNER'
    """
    # BFS flood-fill from Vlad to find all reachable tiles
    reachable = set()
    q = deque([vlad_tile])
    reachable.add(vlad_tile)
    while q:
        node = q.popleft()
        for nb in NAV_GRAPH.get(node, []):
            if nb not in reachable:
                reachable.add(nb)
                q.append(nb)

    if not reachable:
        return None

    pc, pr = player_tile

    def score(t):
        tc, tr = t
        dist_h = abs(tc - pc)
        dist_v = abs(tr - pr)
        if strategy == "HIGH":
            return -tr + dist_h * 0.3       # prefer high rows (small row index) + some horizontal
        elif strategy == "FAR":
            return dist_h + dist_v * 0.5    # prefer max horizontal distance
        else:  # CORNER
            return dist_h + dist_v          # prefer farthest overall corner

    return max(reachable, key=score)


# ── Behaviour-tree mini-library ───────────────────────────────────────────────
_S, _F, _R = "SUCCESS", "FAILURE", "RUNNING"

class _Seq:
    def __init__(self, *ch): self.ch = ch
    def tick(self, c):
        for n in self.ch:
            r = n.tick(c)
            if r != _S: return r
        return _S

class _Sel:
    def __init__(self, *ch): self.ch = ch
    def tick(self, c):
        for n in self.ch:
            r = n.tick(c)
            if r != _F: return r
        return _F

class _Cond:
    def __init__(self, fn): self.fn = fn
    def tick(self, c): return _S if self.fn(c) else _F

class _Act:
    def __init__(self, fn): self.fn = fn
    def tick(self, c): return self.fn(c)


# ── Line of sight ─────────────────────────────────────────────────────────────
def has_line_of_sight(rect_a, rect_b):
    x1, y1 = rect_a.centerx, rect_a.centery
    x2, y2 = rect_b.centerx, rect_b.centery
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist > VISION_RANGE:
        return False
    steps = max(1, int(dist / 6))
    for i in range(1, steps):
        px = int(x1 + (x2 - x1) * i / steps)
        py = int(y1 + (y2 - y1) * i / steps)
        if _is_solid(px // TILE, py // TILE):
            return False
    return True


# ── Situational awareness builder ────────────────────────────────────────────
def build_situation_text(vlad_tile, player_tile, ammo, guard_data=None):
    vc, vr = vlad_tile
    pc, pr = player_tile

    # All tiles reachable from Vlad via BFS flood-fill
    reachable = set()
    q = deque([vlad_tile])
    reachable.add(vlad_tile)
    while q:
        node = q.popleft()
        for nb in NAV_GRAPH.get(node, []):
            if nb not in reachable:
                reachable.add(nb)
                q.append(nb)

    dist  = abs(vc - pc) + abs(vr - pr)
    v_rel = "ABOVE you" if pr < vr else "BELOW you" if pr > vr else "same row"
    h_rel = "to your LEFT" if pc < vc else "to your RIGHT"
    danger = "CRITICAL — assassin almost on top of you!" if dist <= 3 \
             else "CLOSE — act fast!" if dist <= 6 else "manageable distance"

    # Strategic targets
    highest   = min(reachable, key=lambda t: t[1],                         default=None)
    farthest  = max(reachable, key=lambda t: abs(t[0] - pc),               default=None)
    corner    = max(reachable, key=lambda t: abs(t[0]-pc)+abs(t[1]-pr),   default=None)

    def steps(goal):
        if goal is None: return "N/A"
        p = bfs_path(vlad_tile, goal)
        return f"{len(p)-1} steps" if p else "unreachable"

    # Dead ends = tiles with only 1 nav neighbour (traps — avoid when fleeing)
    traps = [(c, r) for (c, r) in reachable
             if len(NAV_GRAPH.get((c, r), [])) <= 1 and (c, r) != vlad_tile]

    fire_note = (f"You have {ammo} fireball(s). FIRE is most useful when assassin "
                 f"is on the SAME row and close." if ammo > 0 else
                 "No fireballs left — do NOT choose FIRE.")

    lines = [
        f"Threat level: {danger}",
        f"Assassin: {dist} tiles away, {v_rel}, {h_rel}",
        "",
        "Reachable escape targets:",
        f"  HIGH   → {highest}  ({steps(highest)})",
        f"  FAR    → {farthest}  ({steps(farthest)})",
        f"  CORNER → {corner}  ({steps(corner)})",
        "",
        f"Dead-end traps to AVOID: {traps[:5] if traps else 'none'}",
        "",
        fire_note,
    ]

    # Guard awareness
    if guard_data:
        alive_guards = [g for g in guard_data if g["alive"]]
        if alive_guards:
            lines += ["", "Your GUARDS (GUARD = deploy them to intercept assassin):"]
            for i, g in enumerate(alive_guards):
                gc, gr  = g["tile"]
                gdist   = abs(gc - pc) + abs(gr - pr)
                lines.append(
                    f"  Guard {i+1}: tile ({gc},{gr}), mode={g['mode']}, "
                    f"{gdist} tiles from assassin — {'on ceiling, will drop & fire' if g['mode']=='ceiling' else 'already on floor, will charge'}"
                )
            lines.append("  Directives: AMBUSH (lure into drop), PINCER (flank both sides), HUNKER (charge), GUARD (immediate drop).")
        else:
            lines += ["", "Guards: all eliminated — do NOT choose GUARD."]

    return "\n".join(lines)


# ── Ollama AI (picks escape strategy) ────────────────────────────────────────
class VladAI:
    STRATEGIES = ("FLEE", "AMBUSH", "PINCER", "HUNKER", "FIRE", "GUARD", "SWAP")
    MAX_LOG    = 5

    def __init__(self):
        self._strategy  = "FAR"
        self._lock      = threading.Lock()
        self._thinking  = False
        self._last_call = 0.0
        self.new_plan   = False          # set True when fresh strategy arrives
        self.log        = []             # list of chat dicts for the side panel

    def strategy(self):
        with self._lock:
            return self._strategy

    def pop_new_plan(self):
        """Return True (and reset flag) if a new plan just arrived."""
        with self._lock:
            flag = self.new_plan
            self.new_plan = False
        return flag

    def get_log(self):
        with self._lock:
            return list(self.log)

    def request(self, vlad_col, vlad_row, player_col, player_row,
                panic=False, ammo=3, guard_data=None, swap_available=False):
        now      = time.time()
        interval = AI_PANIC if panic else AI_INTERVAL
        if self._thinking or (now - self._last_call) < interval:
            return
        self._thinking  = True
        self._last_call = now
        args = (vlad_col, vlad_row, player_col, player_row, ammo,
                guard_data or [], swap_available)
        threading.Thread(target=self._call, args=args, daemon=True).start()

    def _call(self, vc, vr, pc, pr, ammo, guard_data, swap_available):
        chosen = None
        raw    = ""
        try:
            # Build full ASCII map with V and A marked
            map_lines = []
            for row in range(LEVEL_ROWS):
                line = ""
                for col in range(LEVEL_COLS):
                    if row == vr and col == vc:
                        line += 'V'
                    elif row == pr and col == pc:
                        line += 'A'
                    else:
                        line += _tile_at(col, row)
                map_lines.append(line)
            map_str = "\n".join(map_lines)

            situation    = build_situation_text((vc, vr), (pc, pr), ammo, guard_data)
            alive_guards = [g for g in guard_data if g["alive"]]
            fire_opt   = "- FIRE:   launch fireball (best when assassin is on same row, close)\n" if ammo > 0 else ""
            fire_word  = ", FIRE" if ammo > 0 else ""
            guard_opt  = "- GUARD:  drop guards on assassin immediately\n" if alive_guards else ""
            guard_word = ", GUARD" if alive_guards else ""
            swap_opt   = "- SWAP:   teleport into a guard's body — vanish in smoke, reappear elsewhere (ONE USE ONLY)\n" if swap_available and alive_guards else ""
            swap_word  = ", SWAP" if swap_available and alive_guards else ""

            prompt = (
                "You are Vlad the Impaler. An assassin (A) is hunting you.\n"
                "Use the situational data below to choose the BEST tactical directive.\n\n"
                f"=== MAP ===\n{map_str}\n\n"
                f"=== SITUATION ===\n{situation}\n\n"
                "=== YOUR OPTIONS ===\n"
                "- FLEE:   run from the assassin via optimal BFS path\n"
                "- AMBUSH: lure assassin toward ceiling guards, then they drop\n"
                "- PINCER: hold position, signal guards to flank from both sides\n"
                "- HUNKER: retreat to highest platform, guards charge the assassin\n"
                f"{fire_opt}"
                f"{guard_opt}"
                f"{swap_opt}"
                "\nGUIDELINES:\n"
                "  AMBUSH is powerful when guards are on ceiling and assassin is approaching.\n"
                "  PINCER works when you are between two guards.\n"
                "  HUNKER when you need distance and assassin is far.\n"
                "  SWAP when critically threatened and guards are alive — use it to escape.\n"
                "  FLEE when no other option is good.\n"
                f"Reply with exactly ONE word: FLEE, AMBUSH, PINCER, HUNKER{fire_word}{guard_word}{swap_word}"
            )

            payload = json.dumps({
                "model"  : OLLAMA_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {"temperature": 0.3, "num_predict": 10}
            }).encode()

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = json.loads(resp.read()).get("response", "").strip()
                for word in raw.upper().split():
                    if word in self.STRATEGIES:
                        chosen = word
                        break

            if chosen:
                with self._lock:
                    self._strategy = chosen
                    self.new_plan  = True
                    dist = abs(vc-pc) + abs(vr-pr)
                    entry = {
                        "time"    : time.strftime("%H:%M:%S"),
                        "prompt"  : f"Vlad({vc},{vr}) Assassin({pc},{pr}) dist={dist}t",
                        "response": raw[:60],
                        "strategy": chosen,
                    }
                    self.log.append(entry)
                    if len(self.log) > self.MAX_LOG:
                        self.log.pop(0)
        except Exception:
            pass
        finally:
            self._thinking = False


# ── Physics helpers ───────────────────────────────────────────────────────────
def collide_x(rect, vx, tiles):
    for t in tiles:
        if rect.colliderect(t):
            if vx > 0: rect.right = t.left
            elif vx < 0: rect.left = t.right

def collide_y(rect, vy, tiles):
    on_ground = False
    for t in tiles:
        if rect.colliderect(t):
            if vy > 0:
                rect.bottom = t.top
                on_ground = True
            elif vy < 0:
                rect.top = t.bottom
            vy = 0
    return on_ground, vy

def apply_physics(rect, vx, vy, tiles):
    vy = min(vy + GRAVITY, 20)
    rect.x += int(vx)
    collide_x(rect, vx, tiles)
    steps  = max(1, int(abs(vy) // 8))
    step_y = vy / steps
    on_ground = False
    for _ in range(steps):
        rect.y += int(step_y)
        on_ground, vy = collide_y(rect, vy, tiles)
        if on_ground:
            vy = 0
    return vy, on_ground


# ── Player ────────────────────────────────────────────────────────────────────
class Player:
    W, H    = 38, 64
    CRAWL_H = 30          # hitbox height when crouching (~fits in 1-tile gap)

    def __init__(self, x, y, img, crouch_img=None, walk_img=None):
        self.rect       = pygame.Rect(x, y, self.W, self.H)
        self.vx = self.vy = 0
        self.on_ground  = False
        self.img_flip   = False
        self.img        = img
        self.crouch_img = crouch_img
        self.walk_img   = walk_img
        self.daggers    = 0
        self.crouching  = False

    def _can_stand(self):
        """Return True if there is enough vertical space to stand up."""
        extra_top = self.rect.bottom - self.H
        extra_bot = self.rect.top
        if extra_top >= extra_bot:
            return True
        col_l = self.rect.left  // TILE
        col_r = (self.rect.right - 1) // TILE
        for row in range(extra_top // TILE, extra_bot // TILE + 1):
            if _is_solid(col_l, row) or _is_solid(col_r, row):
                return False
        return True

    def handle_input(self, keys):
        self.vx = 0
        want_crouch = keys[pygame.K_DOWN] or keys[pygame.K_s]
        if want_crouch and not self.crouching and self.on_ground:
            self.crouching = True
            bot = self.rect.bottom
            self.rect.height = self.CRAWL_H
            self.rect.bottom = bot
        elif not want_crouch and self.crouching and self._can_stand():
            self.crouching = False
            bot = self.rect.bottom
            self.rect.height = self.H
            self.rect.bottom = bot
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]:
            self.vx = -PLAYER_SPEED; self.img_flip = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.vx =  PLAYER_SPEED; self.img_flip = False
        if (keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]) and self.on_ground and not self.crouching:
            self.vy = JUMP_FORCE; self.on_ground = False

    def update(self, tiles):
        self.vy, self.on_ground = apply_physics(self.rect, self.vx, self.vy, tiles)

    def draw(self, surface, cam):
        r = cam.apply(self.rect)
        if self.crouching:
            img = self.crouch_img
        elif self.vx != 0 and self.walk_img:
            img = self.walk_img
        else:
            img = self.img
        if img:
            surface.blit(pygame.transform.flip(img, self.img_flip, False), r)
        else:
            pygame.draw.rect(surface, GOLD, r)


# ── Fireball ──────────────────────────────────────────────────────────────────
class Fireball:
    SPEED = 2
    R     = 10

    LIFETIME = 360   # 6 seconds at 60 fps

    def __init__(self, x, y, direction):
        self.x      = float(x)
        self.y      = float(y)
        self.vx     = self.SPEED if direction > 0 else -self.SPEED
        self.active = True
        self._tick  = 0
        self._life  = self.LIFETIME

    @property
    def rect(self):
        return pygame.Rect(int(self.x) - self.R, int(self.y) - self.R,
                           self.R * 2, self.R * 2)

    def update(self, solids):
        if not self.active:
            return
        self._life -= 1
        if self._life <= 0:
            self.active = False
            return
        self.x    += self.vx
        self._tick = (self._tick + 1) % 20
        r = self.rect
        for t in solids:
            if r.colliderect(t):
                self.active = False
                return
        if self.x < 0 or self.x > LEVEL_PIXEL_W:
            self.active = False

    def draw(self, surface, cam):
        if not self.active:
            return
        r  = cam.apply(self.rect)
        cx, cy = r.centerx, r.centery
        pulse  = self.R + int(3 * math.sin(self._tick * math.pi / 10))
        # Outer glow
        glow = pygame.Surface((pulse*4, pulse*4), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 80, 0, 90), (pulse*2, pulse*2), pulse*2)
        surface.blit(glow, (cx - pulse*2, cy - pulse*2))
        # Core layers
        pygame.draw.circle(surface, (255, 130, 0), (cx, cy), pulse)
        pygame.draw.circle(surface, (255, 220, 80), (cx, cy), max(1, pulse - 4))
        pygame.draw.circle(surface, (255, 255, 220), (cx, cy), max(1, pulse - 7))


# ── SpatialGrid (broad-phase collision, cell = 3 tiles = 144 px) ──────────────
_SKULL_CELL = TILE * 3

class SpatialGrid:
    def __init__(self):
        self._d = {}

    def _key(self, x, y):
        return int(x) // _SKULL_CELL, int(y) // _SKULL_CELL

    def rebuild(self, balls):
        self._d = {}
        for b in balls:
            k = self._key(b.x, b.y)
            self._d.setdefault(k, []).append(b)

    def nearby(self, x, y):
        ci, cj = self._key(x, y)
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                lst = self._d.get((ci+di, cj+dj))
                if lst:
                    out.extend(lst)
        return out


# ── SkullBall ─────────────────────────────────────────────────────────────────
class SkullBall:
    R      = 14       # radius px
    BOUNCE = 0.62     # restitution vs tiles / walls
    FRIC   = 0.985    # per-frame horizontal friction on floor
    BALL_E = 0.75     # restitution ball–ball / ball–character

    def __init__(self, x, y):
        self.x     = float(x)
        self.y     = float(y)
        self.vx    = random.uniform(-0.5, 0.5)
        self.vy    = 0.0
        self.angle = random.uniform(0, 360)   # degrees, for rolling visual

    # ── tile collision via 4 face probes ─────────────────────────────────────
    def _tile_bounce(self):
        R = self.R
        # bottom
        if _is_solid(int(self.x) // TILE, int(self.y + R) // TILE):
            self.y = (int(self.y + R) // TILE) * TILE - R - 0.01
            if self.vy > 0:
                self.vy = -abs(self.vy) * self.BOUNCE
            self.vx *= self.FRIC
        # top
        if _is_solid(int(self.x) // TILE, int(self.y - R) // TILE):
            self.y = (int(self.y - R) // TILE + 1) * TILE + R + 0.01
            if self.vy < 0:
                self.vy = abs(self.vy) * self.BOUNCE
        # right
        if _is_solid(int(self.x + R) // TILE, int(self.y) // TILE):
            self.x = (int(self.x + R) // TILE) * TILE - R - 0.01
            if self.vx > 0:
                self.vx = -abs(self.vx) * self.BOUNCE
        # left
        if _is_solid(int(self.x - R) // TILE, int(self.y) // TILE):
            self.x = (int(self.x - R) // TILE + 1) * TILE + R + 0.01
            if self.vx < 0:
                self.vx = abs(self.vx) * self.BOUNCE

    # ── circle–circle collision (equal mass) ─────────────────────────────────
    def collide_ball(self, other):
        dx, dy = other.x - self.x, other.y - self.y
        dist   = math.hypot(dx, dy)
        min_d  = self.R + other.R
        if dist == 0 or dist >= min_d:
            return
        nx, ny  = dx / dist, dy / dist
        ov      = (min_d - dist) * 0.5
        self.x  -= nx * ov;  self.y  -= ny * ov
        other.x += nx * ov;  other.y += ny * ov
        rel_vn  = (self.vx - other.vx) * nx + (self.vy - other.vy) * ny
        if rel_vn > 0:
            imp      = rel_vn * self.BALL_E
            self.vx  -= imp * nx;  self.vy  -= imp * ny
            other.vx += imp * nx;  other.vy += imp * ny

    # ── bounce off a character rect ──────────────────────────────────────────
    def collide_rect(self, rect):
        R  = self.R
        cx = max(rect.left, min(self.x, rect.right))
        cy = max(rect.top,  min(self.y, rect.bottom))
        dx, dy = self.x - cx, self.y - cy
        dist   = math.hypot(dx, dy)
        if dist == 0 or dist >= R:
            return
        nx, ny  = dx / dist, dy / dist
        self.x += nx * (R - dist + 0.5)
        self.y += ny * (R - dist + 0.5)
        rel_vn  = self.vx * nx + self.vy * ny
        if rel_vn < 0:
            self.vx -= (1 + self.BALL_E) * rel_vn * nx
            self.vy -= (1 + self.BALL_E) * rel_vn * ny

    # ── per-frame physics step ────────────────────────────────────────────────
    def update(self, skull_grid, player_rect, vlad_rect):
        self.vy = min(self.vy + GRAVITY, 18)
        self.x += self.vx
        self.y += self.vy
        # world bounds
        if self.x - self.R < 0:
            self.x = self.R; self.vx = abs(self.vx) * self.BOUNCE
        if self.x + self.R > LEVEL_PIXEL_W:
            self.x = LEVEL_PIXEL_W - self.R; self.vx = -abs(self.vx) * self.BOUNCE
        if self.y - self.R < 0:
            self.y = self.R; self.vy = abs(self.vy) * self.BOUNCE
        # respawn if fallen out of world
        if self.y > LEVEL_PIXEL_H + 200:
            nodes = list(NAV_GRAPH.keys())
            if nodes:
                col, row = random.choice(nodes)
                self.x, self.y = col * TILE + TILE // 2, row * TILE - self.R
            self.vx = self.vy = 0.0
        self._tile_bounce()
        # rolling rotation: arc-length / radius → degrees
        self.angle += self.vx * (180.0 / (math.pi * self.R))
        # ball–ball (broad-phase via grid)
        for other in skull_grid.nearby(self.x, self.y):
            if other is not self:
                self.collide_ball(other)
        # character collisions
        self.collide_rect(player_rect)
        self.collide_rect(vlad_rect)

    # ── draw skull sprite (rotated for rolling) ───────────────────────────────
    def draw(self, surface, cam):
        sx = int(self.x) - cam.ox
        sy = int(self.y) - cam.oy
        R  = self.R
        if not (-R < sx < GAME_W + R and -R < sy < SCREEN_H + R):
            return
        # draw skull onto small surface so we can rotate it
        sz  = R * 2 + 2
        tmp = pygame.Surface((sz, sz), pygame.SRCALPHA)
        cx  = cy = sz // 2
        pygame.draw.circle(tmp, (170, 165, 150), (cx, cy), R)
        pygame.draw.circle(tmp, ( 90,  85,  75), (cx, cy), R, 1)
        er = max(1, R // 4)
        ex = R * 33 // 100
        ey = R * 18 // 100
        pygame.draw.circle(tmp, (35, 30, 25), (cx - ex, cy - ey), er)
        pygame.draw.circle(tmp, (35, 30, 25), (cx + ex, cy - ey), er)
        pygame.draw.arc(tmp, (35, 30, 25),
                        pygame.Rect(cx - R // 2, cy, R, R // 2),
                        math.pi, 2 * math.pi, 1)
        rotated = pygame.transform.rotate(tmp, -self.angle)
        surface.blit(rotated, rotated.get_rect(center=(sx, sy)))


# ── Vlad ──────────────────────────────────────────────────────────────────────
class Vlad:
    W, H = 38, 64

    def __init__(self, x, y, img):
        self.rect      = pygame.Rect(x, y, self.W, self.H)
        self.vx = self.vy = 0
        self.on_ground = False
        self.img_flip  = False
        self.img       = img
        self.state           = "IDLE"  # IDLE ↔ FLEE
        self.sees_player     = False
        self.ai              = VladAI()
        self.path            = []
        self.target          = None
        self.flash_timer     = 0       # gold glow — new AI plan
        self.fire_flash      = 0       # red glow — firing
        self._last_roam      = 0.0
        self._roam_speed     = 2
        self.ammo            = 3       # fireballs remaining
        self.pending_fireball= None    # set when Vlad fires; main loop collects
        self._escape_start   = None    # time when Vlad lost sight of player
        self.directive   = "FLEE"     # set by LLM
        self._lure_timer = 0          # frames remaining in LURE state
        self.swap_used   = False      # one-time guard swap ability
        self._bt         = self._build_bt()
        self.particles   = []         # particle effects

    def _foot_tile(self):
        """Tile coordinates of the ground Vlad is standing on."""
        col = self.rect.centerx // TILE
        row = (self.rect.bottom - 1) // TILE
        return col, row

    def _replan(self, player, guards=None, panic=False):
        vc, vr = self._foot_tile()
        pc = player.rect.centerx // TILE
        pr = (player.rect.bottom - 1) // TILE
        guard_data = []
        for g in (guards or []):
            gc = g.rect.centerx // TILE
            gr = (g.rect.bottom - 1) // TILE
            guard_data.append({"alive": g.alive, "tile": (gc, gr), "mode": g.mode})
        swap_available = not self.swap_used and any(g.alive for g in (guards or []))
        self.ai.request(vc, vr, pc, pr, panic=panic, ammo=self.ammo,
                        guard_data=guard_data, swap_available=swap_available)
        # Store directive from AI (used by BT each frame)
        self.directive = self.ai.strategy()
        # For movement: map directives to an escape movement target
        move_map = {
            "FLEE": "FAR", "AMBUSH": "HIGH", "PINCER": "FAR",
            "HUNKER": "HIGH", "FIRE": "FAR", "GUARD": "FAR", "SWAP": "FAR",
        }
        move_strategy = move_map.get(self.directive, "FAR")
        target = best_escape_tile((vc, vr), (pc, pr), move_strategy)
        if target and target != (vc, vr):
            self.target = target
            self.path   = bfs_path((vc, vr), target)

    # ── Utility score: should Vlad fire right now? (0.0–1.0) ─────────────────
    def _fire_utility(self, player):
        if self.ammo <= 0 or not self.sees_player:
            return 0.0
        dx = abs(self.rect.centerx - player.rect.centerx)
        dy = abs(self.rect.centery - player.rect.centery)
        row_align = max(0.0, 1.0 - dy / (TILE * 1.5))   # 1 if same row
        proximity = max(0.0, 1.0 - dx / (TILE * 8))      # 1 if very close
        return row_align * 0.7 + proximity * 0.3

    # ── BT action helpers ─────────────────────────────────────────────────────
    def _bt_fire(self, ctx):
        player = ctx["player"]
        direction = 1 if player.rect.centerx > self.rect.centerx else -1
        self.pending_fireball = Fireball(self.rect.centerx, self.rect.centery, direction)
        self.ammo      -= 1
        self.fire_flash = 40
        return _S

    def _bt_lure(self, ctx):
        """Move briefly toward player to draw them toward guards."""
        if self._lure_timer <= 0:
            self._lure_timer = 90   # lure for 1.5 s then re-evaluate
        self._lure_timer -= 1
        player = ctx["player"]
        dx = player.rect.centerx - self.rect.centerx
        self.vx = (VLAD_SPEED * 0.6) if dx > 0 else -(VLAD_SPEED * 0.6)
        if self._lure_timer <= 0:
            self.vx = 0
        return _R

    def _bt_hold(self, ctx):
        self.vx = 0
        return _S

    def _bt_flee(self, ctx):
        self._follow_path(ctx["tiles"], ctx["player"])
        return _R

    # ── Build Vlad's behaviour tree (called once) ─────────────────────────────
    def _build_bt(self):
        return _Sel(
            # Fire: high-utility shot
            _Seq(
                _Cond(lambda c: self._fire_utility(c["player"]) > 0.65),
                _Act(lambda c: self._bt_fire(c)),
            ),
            # AMBUSH: lure player toward ceiling guards
            _Seq(
                _Cond(lambda c: self.directive == "AMBUSH"),
                _Cond(lambda c: any(g.alive and g.mode in ("ceiling", "patrol")
                                    for g in c["guards"])),
                _Act(lambda c: self._bt_lure(c)),
            ),
            # PINCER: hold still while guards flank
            _Seq(
                _Cond(lambda c: self.directive == "PINCER"),
                _Act(lambda c: self._bt_hold(c)),
            ),
            # Default: BFS flee
            _Act(lambda c: self._bt_flee(c)),
        )

    def update(self, tiles, player, guards=None):
        self.sees_player = has_line_of_sight(self.rect, player.rect)

        # FSM transitions
        if self.sees_player and self.state == "IDLE":
            self.state = "FLEE"
            self._escape_start = None

        if self.state == "FLEE":
            if not self.sees_player:
                if self._escape_start is None:
                    self._escape_start = time.time()
                elif time.time() - self._escape_start >= 10.0:
                    self.state = "IDLE"
                    self.path  = []
                    self._last_roam    = 0.0
                    self._escape_start = None
                    self.directive     = "FLEE"
            else:
                self._escape_start = None

        # New plan from AI — update directive and deploy GUARD if needed
        if self.ai.pop_new_plan() and self.flash_timer == 0:
            self.flash_timer  = 50
            self.directive    = self.ai.strategy()
            self.fire_flash   = 0
            # Gold burst — new plan received
            self._emit_burst(
                [(255, 215, 0), (255, 180, 0), (255, 255, 120)], count=22)
            if self.directive == "FIRE" and self.ammo > 0:
                direction = 1 if player.rect.centerx > self.rect.centerx else -1
                self.pending_fireball = Fireball(
                    self.rect.centerx, self.rect.centery, direction)
                self.ammo      -= 1
                self.fire_flash = 40
                # Orange/red burst — firing
                self._emit_burst(
                    [(255, 80, 0), (255, 140, 0), (255, 220, 60)], count=18)
            elif self.directive == "GUARD" and guards:
                for g in guards:
                    if g.alive and g.mode == "ceiling":
                        g.mode = "dropping"
                        g.vy   = 0.5
                        if not g.fired:
                            g._fire(player)
                self.fire_flash = 40
                # Red burst — deploying guards
                self._emit_burst(
                    [(220, 30, 30), (255, 80, 80), (200, 0, 100)], count=18)
            elif self.directive == "SWAP" and not self.swap_used and guards:
                # Pick the alive guard farthest from the player
                target_guard = None
                best_dist    = -1
                for g in guards:
                    if g.alive:
                        d = math.hypot(g.rect.centerx - player.rect.centerx,
                                       g.rect.centery - player.rect.centery)
                        if d > best_dist:
                            best_dist    = d
                            target_guard = g
                if target_guard:
                    old_cx, old_cy = self.rect.centerx, self.rect.centery
                    new_cx, new_cy = target_guard.rect.centerx, target_guard.rect.centery
                    # Teleport Vlad to guard position and guard to Vlad position
                    self.rect.center        = (new_cx, new_cy)
                    target_guard.rect.center = (old_cx, old_cy)
                    target_guard.mode       = "floor"   # guard lands on floor
                    target_guard.vy         = 0.0
                    self.swap_used = True
                    # Big smoke at both locations
                    self._emit_smoke_at(old_cx, old_cy, count=45)
                    self._emit_smoke_at(new_cx, new_cy, count=45)
            elif self.directive in ("AMBUSH", "PINCER", "HUNKER") and guards:
                # Signal guards with the new directive
                for g in guards:
                    g.directive = self.directive

        if self.flash_timer > 0: self.flash_timer -= 1
        if self.fire_flash  > 0: self.fire_flash  -= 1

        ctx = {"tiles": tiles, "player": player, "guards": guards or [], "vlad": self}

        if self.state == "IDLE":
            now = time.time()
            if self.on_ground and (not self.path or now - self._last_roam > ROAM_INTERVAL):
                nodes = list(NAV_GRAPH.keys())
                if nodes:
                    dest = random.choice(nodes)
                    vc, vr = self._foot_tile()
                    self.path = bfs_path((vc, vr), dest)
                    self._last_roam = now
            self._follow_path_roam()
        else:
            dist  = math.hypot(self.rect.centerx - player.rect.centerx,
                               self.rect.centery - player.rect.centery)
            panic = self.sees_player and dist < PANIC_RANGE

            if self.on_ground and (not self.path or panic):
                self._replan(player, guards=guards, panic=panic)

            # BT drives frame-by-frame action selection
            self._bt.tick(ctx)

        if self.vx < 0:   self.img_flip = True
        elif self.vx > 0: self.img_flip = False

        self.vy, self.on_ground = apply_physics(self.rect, self.vx, self.vy, tiles)

    def _follow_path_roam(self):
        """Follow path at patrol speed (no tiles/player needed — just move)."""
        if not self.path:
            self.vx = 0
            return
        next_col, next_row = self.path[0]
        target_cx = next_col * TILE + TILE // 2
        cur_col, cur_row = self._foot_tile()
        if next_row < cur_row and self.on_ground:
            self.vy = VLAD_JUMP
        dx = target_cx - self.rect.centerx
        if abs(dx) > 6:
            self.vx = self._roam_speed if dx > 0 else -self._roam_speed
        else:
            self.vx = 0
        if cur_col == next_col and cur_row == next_row:
            self.path.pop(0)

    def _follow_path(self, tiles, player):
        if not self.path:
            self.vx = 0
            return

        next_col, next_row = self.path[0]
        target_cx = next_col * TILE + TILE // 2   # pixel centre of waypoint tile
        cur_col, cur_row = self._foot_tile()

        # ── Jump if waypoint is above ────────────────────────────────────────
        if next_row < cur_row and self.on_ground:
            self.vy = VLAD_JUMP

        # ── Move horizontally toward waypoint ───────────────────────────────
        dx = target_cx - self.rect.centerx
        if abs(dx) > 6:
            self.vx = VLAD_SPEED if dx > 0 else -VLAD_SPEED
        else:
            self.vx = 0

        # ── Advance waypoint when tile matches ───────────────────────────────
        if cur_col == next_col and cur_row == next_row:
            self.path.pop(0)

    def _emit_burst(self, colors, count=20):
        for _ in range(count):
            spd = random.uniform(1.8, 5.0)
            ang = random.uniform(0, 2 * math.pi)
            self.particles.append(Particle(
                self.rect.centerx, self.rect.centery,
                math.cos(ang) * spd,
                math.sin(ang) * spd,
                random.randint(28, 50),
                random.choice(colors),
            ))

    def _emit_smoke_at(self, x, y, count=40):
        """Large, fast-fading smoke cloud — used for the swap teleport."""
        _smoke = [(210, 210, 210), (180, 180, 185), (230, 225, 220), (160, 160, 165)]
        for _ in range(count):
            spd = random.uniform(1.0, 4.5)
            ang = random.uniform(0, 2 * math.pi)
            # Bias slightly upward (smoke rises)
            vy  = math.sin(ang) * spd - random.uniform(0.5, 2.0)
            self.particles.append(Particle(
                x, y,
                math.cos(ang) * spd,
                vy,
                random.randint(14, 26),      # short life → fast disappear
                random.choice(_smoke),
                max_r=12,                    # large radius → big puff
            ))

    def draw(self, surface, cam, font):
        r = cam.apply(self.rect)

        if self.img:
            sprite = pygame.transform.flip(self.img, self.img_flip, False)
            surface.blit(sprite, (r.x, r.bottom - sprite.get_height()))
        else:
            pygame.draw.rect(surface, RED, r)

        # Update and draw particles (in front of sprite)
        alive = []
        for p in self.particles:
            if p.update():
                p.draw(surface, cam)
                alive.append(p)
        self.particles = alive

        if self.sees_player:
            bang = font.render("!", True, RED)
            surface.blit(bang, (r.centerx - bang.get_width() // 2, r.top - 28))


# ── Particle ──────────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, vx, vy, life, color, max_r=4):
        self.x     = float(x)
        self.y     = float(y)
        self.vx    = vx
        self.vy    = vy
        self.life  = life
        self.mlife = life
        self.color = color   # (r, g, b)
        self.max_r = max_r

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.06
        self.vx *= 0.96
        self.life -= 1
        return self.life > 0

    def draw(self, surface, cam):
        sx = int(self.x) - cam.ox
        sy = int(self.y) - cam.oy
        if not (-4 < sx < GAME_W + 4 and -4 < sy < SCREEN_H + 4):
            return
        r    = max(1, int(self.max_r * self.life / self.mlife))
        frac = self.life / self.mlife
        col  = (int(self.color[0] * frac),
                int(self.color[1] * frac),
                int(self.color[2] * frac))
        pygame.draw.circle(surface, col, (sx, sy), r)


# ── GuardFireball (purple, particle trail) ────────────────────────────────────
_GFB_COLORS = [(255, 80, 200), (200, 50, 255), (255, 140, 50), (255, 240, 80)]

class GuardFireball:
    SPEED = 3.5
    R     = 8

    LIFETIME = 360   # 6 seconds at 60 fps

    def __init__(self, x, y, direction):
        self.x         = float(x)
        self.y         = float(y)
        self.vx        = self.SPEED * (1 if direction > 0 else -1)
        self.active    = True
        self.particles = []
        self._tick     = 0
        self._life     = self.LIFETIME

    @property
    def rect(self):
        return pygame.Rect(int(self.x) - self.R, int(self.y) - self.R,
                           self.R * 2, self.R * 2)

    @property
    def fully_dead(self):
        return not self.active and not self.particles

    def update(self, solids):
        if self.active:
            self._life -= 1
            if self._life <= 0:
                self.active = False
            else:
                # Only emit trail while the fireball is alive
                for _ in range(3):
                    spd = random.uniform(0.4, 1.4)
                    ang = random.uniform(0, 2 * math.pi)
                    self.particles.append(Particle(
                        self.x, self.y,
                        math.cos(ang) * spd - self.vx * 0.25,
                        math.sin(ang) * spd,
                        random.randint(10, 22),
                        random.choice(_GFB_COLORS),
                    ))
        self.particles = [p for p in self.particles if p.update()]

        if not self.active:
            return
        self.x    += self.vx
        self._tick = (self._tick + 1) % 20
        r = self.rect
        for t in solids:
            if r.colliderect(t):
                self.active = False
                return
        if self.x < 0 or self.x > LEVEL_PIXEL_W:
            self.active = False

    def draw(self, surface, cam):
        for p in self.particles:
            p.draw(surface, cam)
        if not self.active:
            return
        sx = int(self.x) - cam.ox
        sy = int(self.y) - cam.oy
        R  = self.R
        pulse = R + int(2 * math.sin(self._tick * math.pi / 10))
        glow  = pygame.Surface((pulse * 4, pulse * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow, (150, 30, 220, 80), (pulse*2, pulse*2), pulse*2)
        surface.blit(glow, (sx - pulse*2, sy - pulse*2))
        pygame.draw.circle(surface, (180,  40, 255), (sx, sy), pulse)
        pygame.draw.circle(surface, (220, 130, 255), (sx, sy), max(1, pulse - 3))
        pygame.draw.circle(surface, (255, 255, 255), (sx, sy), max(1, pulse - 6))


# ── Guard (ceiling-walking, fires one blast) ──────────────────────────────────
_GUARD_SPD_CEIL  = 1.8
_GUARD_SPD_FLOOR = 2.8

class Guard:
    W, H = 24, 44   # smaller than Vlad

    def __init__(self, x, y, img):
        self.rect        = pygame.Rect(x, y, self.W, self.H)
        self.vx          = 0.0
        self.vy          = 0.0
        self.img         = img
        self.alive       = True
        self.fired       = False
        self.pending_fb  = None
        self.sees_player = False
        # "ceiling" → walk upside-down on ceiling
        # "dropping" → free-fall toward floor
        # "floor"    → chase player along floor
        self.mode        = "ceiling"
        self.on_surface  = True
        self.move_dir    = random.choice([-1, 1])
        self._roam_cd    = random.randint(60, 180)
        self.directive   = "FLEE"    # synced from Vlad's AI
        self._alert_cd   = 0         # countdown before dropping
        self._flank_target = None    # pixel x to move to when flanking
        self._floor_bt   = self._build_floor_bt()

    # ── ceiling walk ──────────────────────────────────────────────────────────
    def _update_ceiling(self, tiles, player):
        self._roam_cd -= 1
        if self._roam_cd <= 0:
            self.move_dir = random.choice([-1, 1])
            self._roam_cd = random.randint(80, 200)

        self.rect.x += int(self.move_dir * _GUARD_SPD_CEIL)

        # Wall ahead → turn around
        ahead_col = (self.rect.right if self.move_dir > 0 else self.rect.left - 1) // TILE
        if _is_solid(ahead_col, self.rect.centery // TILE):
            self.move_dir *= -1
            self.rect.x   += int(self.move_dir * _GUARD_SPD_CEIL * 2)

        # Snap top of rect to bottom of ceiling tile
        ceil_row = (self.rect.top - 1) // TILE
        if _is_solid(self.rect.centerx // TILE, ceil_row):
            self.rect.top = (ceil_row + 1) * TILE
        else:
            self.move_dir *= -1   # edge of ceiling → turn

        # Spot player → enter ALERT state (brief wind-up before drop)
        if self.sees_player and self.mode == "ceiling":
            if abs(player.rect.centerx - self.rect.centerx) < TILE * 5:
                self.mode      = "alert"
                self._alert_cd = 35   # ~0.6 s wind-up

    def _update_alert(self, tiles, player):
        """Brief wind-up on ceiling before dropping — telegraphs the attack."""
        # Snap to ceiling while waiting
        ceil_row = (self.rect.top - 1) // TILE
        if _is_solid(self.rect.centerx // TILE, ceil_row):
            self.rect.top = (ceil_row + 1) * TILE
        self._alert_cd -= 1
        if self._alert_cd <= 0:
            self.mode = "dropping"
            self.vy   = 0.5

    # ── free-fall from ceiling ────────────────────────────────────────────────
    def _update_dropping(self, tiles, player):
        self.vy, self.on_surface = apply_physics(
            self.rect, self.vx, self.vy, tiles)
        # Fire when level with the player (within 1.5 tiles vertically)
        if not self.fired and abs(self.rect.centery - player.rect.centery) < TILE * 1.5:
            self._fire(player)
        if self.on_surface:
            # Fire on landing if still haven't (missed the level crossing)
            if not self.fired:
                self._fire(player)
            self.mode = "floor"
            self.vx   = 0.0

    # ── floor chase toward player ─────────────────────────────────────────────
    def _update_floor(self, tiles, player):
        dx = player.rect.centerx - self.rect.centerx
        self.vx = (_GUARD_SPD_FLOOR if dx > 0 else -_GUARD_SPD_FLOOR) if abs(dx) > 10 else 0.0
        self.vy, self.on_surface = apply_physics(
            self.rect, self.vx, self.vy, tiles)

    def _fire(self, player):
        self.fired = True
        direction  = 1 if player.rect.centerx > self.rect.centerx else -1
        self.pending_fb = GuardFireball(
            self.rect.centerx, self.rect.centery, direction)

    # ── Floor BT helpers ──────────────────────────────────────────────────────
    def _bt_flank(self, ctx):
        """Move to a position between player and Vlad (intercept path)."""
        player    = ctx["player"]
        vlad_rect = ctx["vlad_rect"]
        if vlad_rect is None:
            return _F
        if self._flank_target is None:
            self._flank_target = (player.rect.centerx + vlad_rect.centerx) // 2
        dx = self._flank_target - self.rect.centerx
        self.vx = (_GUARD_SPD_FLOOR if dx > 0 else -_GUARD_SPD_FLOOR) if abs(dx) > 12 else 0.0
        if abs(dx) <= 12:
            self._flank_target = None   # reached — recalculate next frame
        return _R

    def _bt_hunt(self, ctx):
        player = ctx["player"]
        dx = player.rect.centerx - self.rect.centerx
        self.vx = (_GUARD_SPD_FLOOR if dx > 0 else -_GUARD_SPD_FLOOR) if abs(dx) > 10 else 0.0
        return _R

    def _bt_hold(self, ctx):
        self.vx = 0.0
        return _S

    def _build_floor_bt(self):
        return _Sel(
            # PINCER: move to flank position between player and Vlad
            _Seq(
                _Cond(lambda c: c["guard"].directive == "PINCER"),
                _Act(lambda c: c["guard"]._bt_flank(c)),
            ),
            # AMBUSH: hold position after landing (Vlad is luring player)
            _Seq(
                _Cond(lambda c: c["guard"].directive == "AMBUSH"),
                _Cond(lambda c: not c["guard"].sees_player),
                _Act(lambda c: c["guard"]._bt_hold(c)),
            ),
            # HUNKER: charge player aggressively
            _Seq(
                _Cond(lambda c: c["guard"].directive == "HUNKER"),
                _Act(lambda c: c["guard"]._bt_hunt(c)),
            ),
            # Default: hunt player
            _Act(lambda c: c["guard"]._bt_hunt(c)),
        )

    def update(self, tiles, player, vlad_rect=None):
        if not self.alive:
            return
        self.sees_player = has_line_of_sight(self.rect, player.rect)
        if self.mode == "ceiling":
            self._update_ceiling(tiles, player)
        elif self.mode == "alert":
            self._update_alert(tiles, player)
        elif self.mode == "dropping":
            self._update_dropping(tiles, player)
        else:
            # BT drives floor behavior
            ctx = {"guard": self, "player": player,
                   "tiles": tiles, "vlad_rect": vlad_rect}
            self._floor_bt.tick(ctx)
            self.vy, self.on_surface = apply_physics(
                self.rect, self.vx, self.vy, tiles)

    def draw(self, surface, cam, font):
        if not self.alive:
            return
        r          = cam.apply(self.rect)
        on_ceiling = (self.mode == "ceiling")
        if self.img:
            sp = pygame.transform.flip(self.img, self.vx < 0, on_ceiling)
            surface.blit(sp, r)
        else:
            pygame.draw.rect(surface, (25, 25, 35), r)
        if self.sees_player:
            bang = font.render("!", True, (220, 50, 50))
            y_off = r.bottom + 4 if on_ceiling else r.top - 24
            surface.blit(bang, (r.centerx - bang.get_width() // 2, y_off))


# ── Camera ────────────────────────────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.ox = self.oy = 0

    def update(self, rect):
        self.ox = max(0, min(rect.centerx - GAME_W // 2, LEVEL_PIXEL_W - GAME_W))
        self.oy = max(0, min(rect.centery - SCREEN_H // 2, LEVEL_PIXEL_H - SCREEN_H))

    def apply(self, rect):
        return rect.move(-self.ox, -self.oy)


# ── Assets ────────────────────────────────────────────────────────────────────
def load_sprite(path, w, h):
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, (w, h))
    except Exception:
        return None

def make_fallback(w, h, colour):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill(colour)
    return s


# ── Level build ───────────────────────────────────────────────────────────────
def build_level():
    solids = []; daggers = []; wall_rects = []; floor_rects = []
    player_start  = (TILE + 8, (LEVEL_ROWS - 2) * TILE - Player.H)
    vlad_start    = None
    guard_starts  = []

    for row, line in enumerate(LEVEL):
        for col, ch in enumerate(line):
            x, y = col * TILE, row * TILE
            r = pygame.Rect(x, y, TILE, TILE)
            if ch == 'W':
                solids.append(r); wall_rects.append(r)
            elif ch == 'F':
                solids.append(r); floor_rects.append(r)
            elif ch == 'G':
                # Guard ceiling tile — solid like W, but records a guard spawn point
                solids.append(r); wall_rects.append(r)
                guard_starts.append((x + 4, (row + 1) * TILE))   # rect.top = bottom of tile
            elif ch == 'D':
                daggers.append(pygame.Rect(x + 12, y + 8, 24, 40))
            elif ch == 'V':
                vlad_start = (x + 4, (row + 1) * TILE - Vlad.H)
            elif ch == 'P':
                player_start = (col * TILE + 4, (row + 1) * TILE - Player.H)

    return solids, daggers, wall_rects, floor_rects, player_start, vlad_start, guard_starts


# ── Skull spawn ───────────────────────────────────────────────────────────────
def spawn_skulls(n=55):
    nodes = list(NAV_GRAPH.keys())
    if not nodes:
        return []
    balls = []
    for _ in range(n):
        col, row = random.choice(nodes)
        x = col * TILE + TILE // 2 + random.uniform(-8, 8)
        y = row * TILE - SkullBall.R - 2
        balls.append(SkullBall(x, y))
    return balls


# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(surface, player, total_daggers, vlad, font):
    panel = pygame.Surface((520, 44), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    surface.blit(panel, (8, 8))
    labels = {"IDLE": "roaming", "FLEE": f"FLEEING! [{vlad.ai.strategy()}]"}
    state_txt = labels.get(vlad.state, vlad.state)
    fire_txt  = f"  🔥 {vlad.ammo}" if vlad.ammo > 0 else "  🔥 OUT"
    txt = font.render(
        f"Daggers: {player.daggers}/{total_daggers}    Vlad: {state_txt}  Fireballs:{vlad.ammo}",
        True, GOLD)
    surface.blit(txt, (14, 16))


CHAT_W = 270

def draw_chat_panel(surface, vlad_ai, font_small):
    """Right-side panel showing Ollama input/output history."""
    log     = vlad_ai.get_log()
    panel_x = GAME_W          # starts right after game area
    panel_h = SCREEN_H

    bg = pygame.Surface((CHAT_W, panel_h))
    bg.fill((18, 10, 8))
    surface.blit(bg, (panel_x, 0))
    pygame.draw.line(surface, GOLD, (panel_x, 0), (panel_x, panel_h), 2)

    px = panel_x + 8   # left margin inside panel

    # Title bar
    title = font_small.render(f"  Vlad AI  —  {OLLAMA_MODEL}", True, GOLD)
    surface.blit(title, (px, 8))
    pygame.draw.line(surface, GOLD, (panel_x + 2, 28), (SCREEN_W - 2, 28), 1)

    if not log:
        hint = font_small.render("Waiting for line-of-sight...", True, (120, 120, 120))
        surface.blit(hint, (px, 40))
        return

    y = 34
    for entry in reversed(log):          # newest first
        ts = font_small.render(entry["time"], True, (120, 120, 120))
        surface.blit(ts, (px, y)); y += 16

        inp = font_small.render("IN:", True, (100, 180, 255))
        surface.blit(inp, (px, y))
        pos = font_small.render(f"  {entry['prompt']}", True, (200, 220, 255))
        surface.blit(pos, (px, y)); y += 16

        # Wrap raw response
        resp_raw = entry["response"]
        words, line_buf, resp_lines = resp_raw.split(), "", []
        for w in words:
            test = (line_buf + " " + w).strip()
            if font_small.size(test)[0] < CHAT_W - 18:
                line_buf = test
            else:
                resp_lines.append(line_buf); line_buf = w
        if line_buf:
            resp_lines.append(line_buf)

        for i, rl in enumerate(resp_lines):
            prefix = "OUT: " if i == 0 else "     "
            colour = (255, 220, 80) if i == 0 else (190, 190, 100)
            surface.blit(font_small.render(prefix + rl, True, colour), (px, y))
            y += 16

        strat_surf = font_small.render(f"→  {entry['strategy']}", True, (80, 255, 120))
        surface.blit(strat_surf, (px, y)); y += 20

        pygame.draw.line(surface, (50, 40, 30), (panel_x + 4, y), (SCREEN_W - 4, y), 1)
        y += 6

        if y > panel_h - 20:
            break


def draw_message(surface, text, font_big, colour=GOLD, reason=""):
    overlay = pygame.Surface((GAME_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))
    msg = font_big.render(text, True, colour)
    surface.blit(msg, msg.get_rect(center=(GAME_W // 2, SCREEN_H // 2)))
    if reason:
        font_reason = pygame.font.SysFont("serif", 26, italic=True)
        rsuf = font_reason.render(reason, True, (220, 180, 180))
        surface.blit(rsuf, rsuf.get_rect(center=(GAME_W // 2, SCREEN_H // 2 + 44)))
    sub = pygame.font.SysFont("serif", 28).render("Press R to restart", True, WHITE)
    surface.blit(sub, sub.get_rect(center=(GAME_W // 2, SCREEN_H // 2 + 80)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock  = pygame.time.Clock()

    level_index = _load_level_index()
    # Start at whatever was loaded at boot, or first entry
    level_idx = 0
    for i, e in enumerate(level_index):
        if os.path.abspath(e["file"]) == os.path.abspath(CURRENT_LEVEL_PATH):
            level_idx = i
            break

    def _update_caption():
        entry = level_index[level_idx]
        pygame.display.set_caption(
            f"The Assassin  |  {entry['name']}  "
            f"({level_idx + 1}/{len(level_index)})  "
            f"[ ] prev/next  R restart"
        )
    _update_caption()

    win_timer = 0   # counts down after win before auto-advancing

    BASE = os.path.join(os.path.dirname(__file__), "sprites")
    player_img = load_sprite(os.path.join(BASE, "assassin.png"), Player.W, Player.H)
    if player_img:
        player_img = pygame.transform.flip(player_img, True, False)
    walk_img   = load_sprite(os.path.join(BASE, "walking.png"), Player.W, Player.H)
    crouch_img = load_sprite(os.path.join(BASE, "crouch.png"), Player.W, Player.CRAWL_H)
    vlad_img   = load_sprite(os.path.join(BASE, "vlad.png"),       Vlad.W, Vlad.H * 2)
    if vlad_img:
        vlad_img = pygame.transform.flip(vlad_img, True, False)
    # Guard: same sprite as Vlad, scaled smaller, tinted near-black
    if vlad_img:
        _gb = pygame.transform.smoothscale(vlad_img, (Guard.W, Guard.H))
        _gt = pygame.Surface(_gb.get_size(), pygame.SRCALPHA)
        _gt.fill((170, 170, 175, 255))
        _gb.blit(_gt, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        guard_img = _gb
    else:
        guard_img = None
    dagger_img = load_sprite(os.path.join(BASE, "dagger.png"),     24, 40)
    wall_img   = load_sprite(os.path.join(BASE, "wall_tile.png"),  TILE, TILE)
    floor_img  = load_sprite(os.path.join(BASE, "floor_tile.png"), TILE, TILE)
    if not wall_img:  wall_img  = make_fallback(TILE, TILE, BROWN)
    if not floor_img: floor_img = make_fallback(TILE, TILE, (60, 40, 20))

    font       = pygame.font.SysFont("serif", 22)
    font_big   = pygame.font.SysFont("serif", 52, bold=True)
    font_bang  = pygame.font.SysFont("serif", 28, bold=True)
    font_small = pygame.font.SysFont("monospace", 13)

    def reset():
        solids, daggers, wall_rects, floor_rects, pstart, vstart, gstarts = build_level()
        player     = Player(*pstart, player_img, crouch_img, walk_img)
        vlad       = Vlad(*(vstart or (LEVEL_COLS // 2 * TILE, TILE * 2)), vlad_img)
        skulls     = spawn_skulls()
        skull_grid = SpatialGrid()
        # Spawn guards from G tiles in level.txt
        guards_list = []
        for i, (gx, gy) in enumerate(gstarts):
            g = Guard(gx, gy, guard_img)
            if i % 2 == 1:
                g.move_dir = -1   # alternate facing direction
            guards_list.append(g)
        return (player, vlad, solids, daggers, wall_rects, floor_rects,
                [], guards_list, [], skulls, skull_grid)

    (player, vlad, solids, daggers, wall_rects, floor_rects,
     fireballs, guards, guard_fbs, skulls, skull_grid) = reset()
    total_daggers = len(daggers)
    cam          = Camera()
    state        = "play"
    death_reason = ""
    death_timer  = 0    # frames to wait before showing death overlay

    bg = pygame.Surface((GAME_W, SCREEN_H))
    for y in range(SCREEN_H):
        t = y / SCREEN_H
        pygame.draw.line(bg, (int(10+20*t), int(5+10*t), int(15+20*t)), (0, y), (GAME_W, y))

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    (player, vlad, solids, daggers, wall_rects, floor_rects,
                     fireballs, guards, guard_fbs, skulls, skull_grid) = reset()
                    total_daggers = len(daggers)
                    state        = "play"
                    death_reason = ""
                    death_timer  = 0
                # ] → next level
                if event.key == pygame.K_RIGHTBRACKET and len(level_index) > 1:
                    level_idx = (level_idx + 1) % len(level_index)
                    switch_level(level_index[level_idx]["file"])
                    (player, vlad, solids, daggers, wall_rects, floor_rects,
                     fireballs, guards, guard_fbs, skulls, skull_grid) = reset()
                    total_daggers = len(daggers)
                    state = "play"; death_reason = ""; death_timer = 0
                    _update_caption()
                # [ → previous level
                if event.key == pygame.K_LEFTBRACKET and len(level_index) > 1:
                    level_idx = (level_idx - 1) % len(level_index)
                    switch_level(level_index[level_idx]["file"])
                    (player, vlad, solids, daggers, wall_rects, floor_rects,
                     fireballs, guards, guard_fbs, skulls, skull_grid) = reset()
                    total_daggers = len(daggers)
                    state = "play"; death_reason = ""; death_timer = 0
                    _update_caption()

        touching_early = False

        if state == "play":
            keys = pygame.key.get_pressed()
            player.handle_input(keys)
            player.update(solids)
            vlad.update(solids, player, guards)
            cam.update(player.rect)

            # Skull ball physics
            skull_grid.rebuild(skulls)
            for sk in skulls:
                sk.update(skull_grid, player.rect, vlad.rect)

            # Guards update + collect pending fireballs
            for g in guards:
                g.update(solids, player, vlad_rect=vlad.rect)
                if g.pending_fb:
                    guard_fbs.append(g.pending_fb)
                    g.pending_fb = None

            # Guard fireballs
            for gfb in guard_fbs[:]:
                gfb.update(solids)
                if gfb.fully_dead:
                    guard_fbs.remove(gfb)
                elif state == "play" and gfb.active and player.rect.colliderect(gfb.rect):
                    state = "dead"; death_timer = 150
                    death_reason = "Struck by a guard's fireball"

            # Guard ↔ player contact
            # Guards are only dangerous when actively attacking (dropping or hunting).
            # On ceiling / alert they are not yet a threat.
            for g in guards:
                if state == "play" and g.alive and player.rect.colliderect(g.rect):
                    if player.daggers >= total_daggers:
                        g.alive = False          # player kills guard with daggers
                    elif g.mode in ("dropping", "floor"):
                        state = "dead"; death_timer = 150
                        death_reason = ("Crushed by a guard dropping from the ceiling"
                                        if g.mode == "dropping"
                                        else "Cut down by Vlad's guard")

            # Collect pending fireball from Vlad
            if vlad.pending_fireball:
                fireballs.append(vlad.pending_fireball)
                vlad.pending_fireball = None

            # Update fireballs
            for fb in fireballs[:]:
                fb.update(solids)
                if not fb.active:
                    fireballs.remove(fb)
                elif state == "play" and player.rect.colliderect(fb.rect):
                    state = "dead"; death_timer = 150
                    death_reason = "Burned alive by Vlad's fireball"

            for d in daggers[:]:
                if player.rect.colliderect(d):
                    daggers.remove(d); player.daggers += 1

            if player.rect.colliderect(vlad.rect):
                if player.daggers >= total_daggers:
                    state = "win"; win_timer = 180
                else:
                    touching_early = True

            if state == "play" and player.rect.bottom > LEVEL_PIXEL_H + TILE:
                state = "dead"; death_timer = 60
                death_reason = "Fell into the abyss"

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.blit(bg, (0, 0))
        for r in wall_rects:  screen.blit(wall_img,  cam.apply(r))
        for r in floor_rects: screen.blit(floor_img, cam.apply(r))

        for d in daggers:
            sr = cam.apply(d)
            if dagger_img: screen.blit(dagger_img, sr)
            else:
                pygame.draw.polygon(screen, (200, 200, 255),
                    [(sr.centerx, sr.top), (sr.centerx-5, sr.bottom), (sr.centerx+5, sr.bottom)])

        for fb in fireballs:
            fb.draw(screen, cam)

        for gfb in guard_fbs:
            gfb.draw(screen, cam)

        for sk in skulls:
            sk.draw(screen, cam)

        for g in guards:
            g.draw(screen, cam, font_bang)

        vlad.draw(screen, cam, font_bang)
        player.draw(screen, cam)
        draw_hud(screen, player, total_daggers, vlad, font)
        draw_chat_panel(screen, vlad.ai, font_small)

        if touching_early:
            warn = font.render(
                f"Collect all daggers first! ({player.daggers}/{total_daggers})", True, RED)
            screen.blit(warn, warn.get_rect(center=(SCREEN_W // 2, 80)))

        if state == "win":
            next_idx = level_idx + 1
            if next_idx < len(level_index):
                # count down 180 frames (~3 s) then auto-advance
                if win_timer > 0:
                    win_timer -= 1
                else:
                    level_idx = next_idx
                    switch_level(level_index[level_idx]["file"])
                    (player, vlad, solids, daggers, wall_rects, floor_rects,
                     fireballs, guards, guard_fbs, skulls, skull_grid) = reset()
                    total_daggers = len(daggers)
                    state = "play"; death_reason = ""; death_timer = 0; win_timer = 0
                    _update_caption()
                secs = (win_timer + 59) // 60
                next_name = level_index[next_idx]["name"]
                draw_message(screen, "VLAD IS DEAD — MISSION COMPLETE!", font_big, GOLD,
                             reason=f"Next: {next_name}  ({secs}s)")
            else:
                draw_message(screen, "CAMPAIGN COMPLETE — VLAD IS DEAD!", font_big, GOLD,
                             reason="All levels finished!")
        elif state == "dead":
            if death_timer > 0:
                death_timer -= 1
            else:
                draw_message(screen, "YOU FELL INTO THE DARKNESS...", font_big, RED,
                             reason=death_reason)

        pygame.display.flip()


# ── Bootstrap: load initial level ─────────────────────────────────────────────
_initial = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(_BASE_DIR, "level.txt"))
switch_level(_initial)

if __name__ == "__main__":
    main()
