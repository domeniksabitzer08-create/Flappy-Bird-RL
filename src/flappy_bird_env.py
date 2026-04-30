import random
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
        self.screen_resolution = (450, 800) # 9:16
        # Game-Design
        self.max_obs_height = 350
        self.min_obs_height = self.screen_resolution[1]-100
        self.obstacles_distance = 300
        self.obstacle_velocity = Vector2D(100, 0)
        self.pipe_height_distance = 300 - (self.difficulty*10)
        # Rendering
        self.player_render_color = (255,165,0)
        # Assign screen rendering is needed
        if render:
            self.screen = pygame.display.set_mode(self.screen_resolution)
        # start procedure
        self.player = None
        self.obstacles = []
        # Starting the logic
        self.start()

    def reset(self):
        self.player = None
        self.obstacles = []
        self.start()

    def start(self):
        # starting code
        self.player = Player(Vector2D(180, 300), Vector2D(0, -500), 15, self)
        # init first pipe
        self.obstacles.append(Obstacle(self.player, Vector2D(self.screen_resolution[0]-50, self.screen_resolution[1]/2 + self.pipe_height_distance/2),Vector2D(self.screen_resolution[0]-50, self.screen_resolution[1]/2 - self.screen_resolution[1] - self.pipe_height_distance/2), self.obstacle_velocity))

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
            # remove obstacle if it is outside the screen
            if obstacle.check_outside_border():
                self.obstacles.pop(0)

        # spawn new obstacle
        self.spawn_obstacle()
        # check for game over
        if self.check_for_game_over():
            self.reset()

    def spawn_obstacle(self):
        """spawns an obstacle if necessary"""
        if len(self.obstacles) < (self.screen_resolution[0] / self.obstacles_distance) +1:
            height = random.randint(self.max_obs_height,self.min_obs_height)
            spawn_pos_x = self.obstacles[len(self.obstacles)-1].pos_1.x+ self.obstacles_distance
            obstacle = Obstacle(self.player, Vector2D(spawn_pos_x , height), Vector2D(spawn_pos_x , height - self.screen_resolution[1] - self.pipe_height_distance ), self.obstacle_velocity)
            self.obstacles.append(obstacle)

    def check_for_game_over(self):
        """return True if the player is dead"""
        for obstacle in self.obstacles:
            if self.check_player_collision(self.player, obstacle):
                return True
            if self.check_border_collision(self.player):
                return True

    def check_player_collision(self, player: "Player", obstacle: "Obstacle") -> bool:
        """Return True if player collides with obstacle"""
        # Find the closest point of the react from the player
        obj_1 = obstacle.pipe_1
        obj_2 = obstacle.pipe_2
        if self.check_circle_react_collision(player, obj_1) or self.check_circle_react_collision(player, obj_2):
            return True
        else:
            return False

    def check_border_collision(self, player: "Player") -> bool:
        """return True if player is outside the screen"""
        if player.pos.y + player.collision_radius > self.screen_resolution[1] or player.pos.y - player.collision_radius < 0 :
            return True
        else:
            return False

    @staticmethod
    def check_circle_react_collision( player: "Player", pipe: pygame.rect) -> bool:
        """Return True if player collides with a pipe"""
        player_rect = pygame.Rect(player.pos.x - player.collision_radius, player.pos.y - player.collision_radius, player.collision_radius*2 , player.collision_radius*2 )
        is_colliding = player_rect.colliderect(pipe)
        return is_colliding

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
        self.collision_radius = 30
        # delta time
        self.clock = pygame.time.Clock()
        self.delta_time = 0
        # Rendering
        self.rendering_radius = 30

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
        self.pipe_collision_height = player.env.screen_resolution[1]
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
        pygame.draw.rect(screen, (255,0,0), self.pipe_1)
        pygame.draw.rect(screen, self.color, self.pipe_2)

    def check_outside_border(self):
        """returns true if the pipes are outside the screen"""
        if self.pos_1.x+self.pipe_collision_width < 0:
            return True
        else:
            return False
















