import pygame
import sys
import os

# ── Constants ──────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1024, 600
FPS = 60
GRAVITY = 0.55
JUMP_FORCE = -14
PLAYER_SPEED = 4
TILE = 48  # tile size in pixels

# ── Colours (fallback if sprites fail) ────────────────────────────────────────
BLACK  = (0,   0,   0  )
WHITE  = (255, 255, 255)
GOLD   = (218, 165, 32 )
RED    = (180, 30,  30 )
DARK   = (20,  12,  5  )
BROWN  = (90,  55,  20 )

# ── Load level from level.txt ──────────────────────────────────────────────────
def _load_level(path):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or line.strip() == "":
                continue
            lines.append(line)
    return lines

LEVEL = _load_level(os.path.join(os.path.dirname(__file__), "level.txt"))

LEVEL_COLS = max(len(r) for r in LEVEL)
LEVEL_ROWS = len(LEVEL)
LEVEL_PIXEL_W = LEVEL_COLS * TILE
LEVEL_PIXEL_H = LEVEL_ROWS * TILE

# Locate P in map for player start; fall back to (1,bottom-2)
def _find_player_start():
    for row, line in enumerate(LEVEL):
        for col, ch in enumerate(line):
            if ch == 'P':
                # place feet at bottom of the P tile
                return (col * TILE + 4, (row + 1) * TILE - 64)
    return (TILE + 8, (LEVEL_ROWS - 2) * TILE - 64)

PLAYER_START = _find_player_start()


def load_sprite(path, w, h):
    """Load image and scale; return None on failure."""
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, (w, h))
    except Exception:
        return None


def make_fallback(w, h, colour):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill(colour)
    return surf


class Camera:
    def __init__(self):
        self.offset_x = 0
        self.offset_y = 0

    def update(self, target_rect):
        self.offset_x = target_rect.centerx - SCREEN_W // 2
        self.offset_y = target_rect.centery - SCREEN_H // 2
        self.offset_x = max(0, min(self.offset_x, LEVEL_PIXEL_W - SCREEN_W))
        self.offset_y = max(0, min(self.offset_y, LEVEL_PIXEL_H - SCREEN_H))

    def apply(self, rect):
        return rect.move(-self.offset_x, -self.offset_y)


class Player:
    W, H = 38, 64

    def __init__(self, x, y, img):
        self.rect = pygame.Rect(x, y, self.W, self.H)
        self.vx = 0
        self.vy = 0
        self.on_ground = False
        self.img = img
        self.img_flip = False   # facing left?
        self.daggers = 0

    def handle_input(self, keys):
        self.vx = 0
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]:
            self.vx = -PLAYER_SPEED
            self.img_flip = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.vx = PLAYER_SPEED
            self.img_flip = False
        if (keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]) and self.on_ground:
            self.vy = JUMP_FORCE
            self.on_ground = False

    def update(self, tiles):
        self.vy = min(self.vy + GRAVITY, 20)  # cap fall speed to prevent tunneling

        # Horizontal move + collision
        self.rect.x += self.vx
        self._collide_x(tiles)

        # Vertical move + collision (step in small increments to avoid tunneling)
        steps = max(1, int(abs(self.vy) // 8))
        step_y = self.vy / steps
        self.on_ground = False
        for _ in range(steps):
            self.rect.y += int(step_y)
            self._collide_y(tiles)

    def _collide_x(self, tiles):
        for t in tiles:
            if self.rect.colliderect(t):
                if self.vx > 0:
                    self.rect.right = t.left
                elif self.vx < 0:
                    self.rect.left = t.right

    def _collide_y(self, tiles):
        for t in tiles:
            if self.rect.colliderect(t):
                if self.vy > 0:
                    self.rect.bottom = t.top
                    self.on_ground = True
                elif self.vy < 0:
                    self.rect.top = t.bottom
                self.vy = 0

    def draw(self, surface, cam, img):
        screen_rect = cam.apply(self.rect)
        if img:
            sprite = pygame.transform.flip(img, self.img_flip, False)
            surface.blit(sprite, screen_rect)
        else:
            pygame.draw.rect(surface, GOLD, screen_rect)


def build_level(wall_img, floor_img):
    """Return (solid_rects, dagger_rects, vlad_rect, wall_surf_list, floor_surf_list)."""
    solids   = []
    daggers  = []
    vlad_pos = None
    wall_draws  = []
    floor_draws = []

    for row, line in enumerate(LEVEL):
        for col, ch in enumerate(line):
            x, y = col * TILE, row * TILE
            r = pygame.Rect(x, y, TILE, TILE)
            if ch == 'W':
                solids.append(r)
                wall_draws.append(r)
            elif ch == 'F':
                solids.append(r)
                floor_draws.append(r)
            elif ch == 'D':
                daggers.append(pygame.Rect(x + 12, y + 8, 24, 40))
            elif ch == 'V':
                vlad_pos = (x, y)
            # P = player start, treated as open air

    return solids, daggers, vlad_pos, wall_draws, floor_draws


def draw_hud(surface, player, total_daggers, font):
    # Dark panel
    panel = pygame.Surface((320, 44), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    surface.blit(panel, (8, 8))
    txt = font.render(
        f"Silver Daggers: {player.daggers}/{total_daggers}   (reach Vlad when full)",
        True, GOLD
    )
    surface.blit(txt, (14, 16))


def draw_message(surface, text, font_big, colour=GOLD):
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))
    msg = font_big.render(text, True, colour)
    surface.blit(msg, msg.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2)))
    sub = pygame.font.SysFont("serif", 28).render("Press R to restart", True, WHITE)
    surface.blit(sub, sub.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 60)))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("The Assassin — Kill Vlad the Impaler")
    clock = pygame.time.Clock()

    # ── Load sprites ───────────────────────────────────────────────────────────
    BASE = os.path.join(os.path.dirname(__file__), "sprites")

    player_img = load_sprite(os.path.join(BASE, "assassin.png"),  Player.W, Player.H)
    if player_img:
        player_img = pygame.transform.flip(player_img, True, False)
    vlad_img   = load_sprite(os.path.join(BASE, "vlad.png"),       TILE,     TILE * 2)
    dagger_img = load_sprite(os.path.join(BASE, "dagger.png"),     24,       40)
    wall_img   = load_sprite(os.path.join(BASE, "wall_tile.png"),  TILE,     TILE)
    floor_img  = load_sprite(os.path.join(BASE, "floor_tile.png"), TILE,     TILE)

    # Fallbacks
    if not wall_img:  wall_img  = make_fallback(TILE, TILE, BROWN)
    if not floor_img: floor_img = make_fallback(TILE, TILE, (60, 40, 20))

    # ── Fonts ──────────────────────────────────────────────────────────────────
    font     = pygame.font.SysFont("serif", 22)
    font_big = pygame.font.SysFont("serif", 52, bold=True)

    # ── Build level ────────────────────────────────────────────────────────────
    def reset():
        solids, daggers, vlad_pos, wall_draws, floor_draws = build_level(wall_img, floor_img)
        player = Player(*PLAYER_START, player_img)
        cam    = Camera()
        vlad_rect = pygame.Rect(vlad_pos[0], vlad_pos[1] - TILE, TILE, TILE * 2) if vlad_pos else None
        return player, solids, daggers, vlad_rect, wall_draws, floor_draws

    player, solids, daggers, vlad_rect, wall_draws, floor_draws = reset()
    total_daggers = len(daggers)

    state = "play"  # play | win | dead

    # ── Background gradient ────────────────────────────────────────────────────
    bg = pygame.Surface((SCREEN_W, SCREEN_H))
    for y in range(SCREEN_H):
        t = y / SCREEN_H
        r = int(10 + 20 * t)
        g = int(5  + 10 * t)
        b = int(15 + 20 * t)
        pygame.draw.line(bg, (r, g, b), (0, y), (SCREEN_W, y))

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    player, solids, daggers, vlad_rect, wall_draws, floor_draws = reset()
                    total_daggers = len(daggers)
                    state = "play"
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

        if state == "play":
            keys = pygame.key.get_pressed()
            player.handle_input(keys)
            player.update(solids)

            # Camera
            cam = Camera()
            cam.update(player.rect)

            # Collect daggers
            for d in daggers[:]:
                if player.rect.colliderect(d):
                    daggers.remove(d)
                    player.daggers += 1

            # Win condition: touch Vlad with all daggers
            if vlad_rect and player.rect.colliderect(vlad_rect):
                if player.daggers >= total_daggers:
                    state = "win"

            # Fall off world = dead
            if player.rect.top > LEVEL_PIXEL_H + 100:
                state = "dead"

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.blit(bg, (0, 0))

        # Tiles
        for r in wall_draws:
            sr = cam.apply(r)
            screen.blit(wall_img, sr)

        for r in floor_draws:
            sr = cam.apply(r)
            screen.blit(floor_img, sr)

        # Daggers
        for d in daggers:
            sr = cam.apply(d)
            if dagger_img:
                screen.blit(dagger_img, sr)
            else:
                pygame.draw.polygon(screen, (200, 200, 255), [
                    (sr.centerx, sr.top),
                    (sr.centerx - 5, sr.bottom),
                    (sr.centerx + 5, sr.bottom)
                ])

        # Vlad
        if vlad_rect:
            sr = cam.apply(vlad_rect)
            if vlad_img:
                screen.blit(vlad_img, sr)
            else:
                pygame.draw.rect(screen, RED, sr)

        # Player
        player.draw(screen, cam, player_img)

        # HUD
        draw_hud(screen, player, total_daggers, font)

        if state == "win":
            draw_message(screen, "VLAD IS DEAD — MISSION COMPLETE!", font_big, GOLD)
        elif state == "dead":
            draw_message(screen, "YOU FELL INTO THE DARKNESS...", font_big, RED)

        pygame.display.flip()


if __name__ == "__main__":
    main()
