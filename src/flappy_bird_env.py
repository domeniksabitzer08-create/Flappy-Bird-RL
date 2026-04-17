import pygame
from dataclasses import dataclass
import math

from pygame.draw import circle
from pygame.examples.scrap_clipboard import screen
from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares


### VECTOR CLASS ###
@dataclass(frozen=True)
class Vector2D:
        x: float
        y: float
        def __add__(self, other):
            # Vector + Vector
            if isinstance(other, Vector2D):
                return Vector2D(self.x + other.x, self.y + other.y)
            else:
                return NotImplemented
        def __sub__(self, other):
            # Vector - Vector
            if isinstance(other, Vector2D):
                return Vector2D(self.x - other.x, self.y - other.y)
            else:
                return NotImplemented
        def __mul__(self, other):
            if isinstance(other, (int, float)):
                # Vector * Number
                return Vector2D(self.x * other, self.y * other)
            else:
                return NotImplemented
        def __truediv__(self, other):
            # Vector / Number
            if isinstance(other, (int, float)):
                return Vector2D(self.x / other, self.y / other)
            else:
                return NotImplemented
        def __eq__(self, other):
            if isinstance(other, Vector2D):
                return self.x == other.x and self.y == other.y
            else:
                return NotImplemented
        def magnitude(self):
            return math.sqrt(self.x * self.x + self.y * self.y)

# Vector directions
Vector2D.left = Vector2D(-1, 0)
Vector2D.right = Vector2D(1, 0)
Vector2D.up = Vector2D(0, -1)
Vector2D.down = Vector2D(0, 1)


class FlappyBirdEnv:
    def __init__(self, difficulty: float, render: bool = False):
        # Overall
        self.difficulty = difficulty
        self.render = render
        self.fps = 60
        # Rendering
        self.player_render_color = (255,165,0)
        # Assign screen rendering is needed
        if render:
            self.screen = pygame.display.set_mode((800, 600))
        # start procedure
        self.player = None

        # Starting the logic
        self.start()

    def start(self):
        # starting code
        self.player = Player(Vector2D(180, 300), Vector2D(0, 10), 1, self)
        running = True
        while running:
            if self.render:
                running = not self.check_for_closing_game()
            self.loop()

    def loop(self):
        if self.render:
            self.player.render(self.screen, self.player_render_color)
            pygame.display.update()

    @staticmethod
    def check_for_closing_game() -> bool:
        """returns True if the window closing button was pressed"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

class Player:
    def __init__(self, start_pos : Vector2D, up_force : Vector2D, mass: float, env: "FlappyBirdEnv"):
        # Environment
        self.env = env
        # Pos Force and Velocity
        self.pos = start_pos
        self.up_force = up_force
        self.velocity = Vector2D(0, 0)
        self.mass = mass
        # Constants
        self.gravity_const = 9.81
        # Collider
        self.collision_radius = 4
        # delta time
        self.clock = pygame.time.Clock()
        self.delta_time = 0
        # Rendering
        self.rendering_radius = 40

    def apply_gravity(self):
        self.velocity += self.gravity_const * self.delta_time

    def calculate_delta_time(self):
        self.delta_time = self.clock.tick(self.env.fps)/1000

    def render(self, screen: pygame.Surface, color = (255,0,0)):
        pygame.draw.circle(screen, color, (self.pos.x, self.pos.y), self.rendering_radius)












