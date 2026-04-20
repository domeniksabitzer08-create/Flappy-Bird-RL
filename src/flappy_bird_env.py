import sys

import pygame
from dataclasses import dataclass
import math




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
        self.obstacles = []
        # Starting the logic
        self.start()

    def start(self):
        # starting code
        self.player = Player(Vector2D(180, 300), Vector2D(0, -500), 15, self)
        # init first pipe
        self.obstacles.append(Obstacle(self.player, Vector2D(400,000),Vector2D(400,400), Vector2D(100,0)))

    def step(self, action: int):
        self.player.update(action)
        if self.render:
            self.screen.fill((0,0,0))
            self.player.render(self.screen, self.player_render_color)
            for obstacle in self.obstacles:
                obstacle.render(self.screen)
            pygame.display.update()
        for obstacle in self.obstacles:
            obstacle.update()


    @staticmethod
    def get_user_input():
        """returns an action depending on user input and closes the game if quit button was pressed"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return 1
        return 0

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
        self.gravity_const = 20
        # Collider
        self.collision_radius = 4
        # delta time
        self.clock = pygame.time.Clock()
        self.delta_time = 0
        # Rendering
        self.rendering_radius = 40

    def update(self, action: int):
        force = self.up_force * action
        self.physics_update(force)

    def physics_update(self, force):
        self.calculate_delta_time()
        self.apply_gravity()
        self.apply_force(force)
        # update position depending on the velocity
        self.pos += self.velocity

    def apply_gravity(self):
        self.velocity += Vector2D(0,self.gravity_const * self.delta_time)

    def apply_force(self, force: Vector2D):
        if force != Vector2D(0, 0):
            self.velocity = force  * self.delta_time

    def calculate_delta_time(self):
        self.delta_time = self.clock.tick(self.env.fps)/1000

    def render(self, screen: pygame.Surface, color = (255,0,0)):
        pygame.draw.circle(screen, color, (self.pos.x, self.pos.y), self.rendering_radius)


class Obstacle:
    def __init__(self, player: Player ,start_pos_1:Vector2D, start_pos_2:Vector2D, velocity:Vector2D):
        self.player = player
        self.pos_1 = start_pos_1
        self.pos_2 = start_pos_2
        self.velocity = velocity

        self.pipe_collision_width = 100
        self.pipe_collision_height = 200
        self.pipe_1 = pygame.Rect(self.pos_1.x, self.pos_1.y, self.pipe_collision_width, self.pipe_collision_height)
        self.pipe_2 = pygame.Rect(self.pos_2.x, self.pos_2.y, self.pipe_collision_width, self.pipe_collision_height)
        # rendering
        self.color = (0,190,0)

    def update(self):
        # move
        self.pos_1 -= self.velocity * self.player.delta_time
        self.pos_2 -= self.velocity * self.player.delta_time
        self.pipe_1 = pygame.Rect(self.pos_1.x, self.pos_1.y, self.pipe_collision_width, self.pipe_collision_height)
        self.pipe_2 = pygame.Rect(self.pos_2.x, self.pos_2.y, self.pipe_collision_width, self.pipe_collision_height)

    def render(self, screen: pygame.Surface):
        pygame.draw.rect(screen, self.color, self.pipe_1)
        pygame.draw.rect(screen, self.color, self.pipe_2)

    def check_outside_border(self):
        """returns true if the pipes are outside the screen"""
        if self.pos_1.x < 0:
            return True
        else:
            return False



class Pipe:
    def __init__(self, start_pos: Vector2D, velocity: Vector2D ):
        pass












