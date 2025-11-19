import pygame
from random import randint
import numpy as np


class FlappyBirdEnv:
    def __init__(self, render_mode=True):
        pygame.init()
        self.WIDTH, self.HEIGHT = 400, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT)) if render_mode else None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("sans", 20)
        self.render_mode = render_mode

        # Colors
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.PINK = (222, 165, 164)
        
        # Game parameters - EASY SETTINGS
        self.TUBE_WIDTH = 30
        self.TUBE_GAP = 400           # Large gap
        self.GRAVITY = 0.22           # Light gravity
        self.JUMP_STRENGTH = -5.0     # Jump strength
        self.TUBE_VELOCITY = 0.8      # Slow tubes
        self.BIRD_X = 50
        self.BIRD_WIDTH = 35
        self.BIRD_HEIGHT = 35

        self.reset()

    def reset(self):
        """Reset game and return initial state"""
        self.Bird_y = 300
        self.bird_vel = 0
        self.score = 0
        self.done = False

        # Tubes start farther and in easier range
        self.tubes = [
            {"x": 700, "height": randint(150, 300), "passed": False},
            {"x": 1050, "height": randint(150, 300), "passed": False},
            {"x": 1400, "height": randint(150, 300), "passed": False},
        ]
        return self.get_state()

    def get_state(self):
        """Return state: [velocity, horizontal_dist, vertical_dist]"""
        y, vy = self.Bird_y, self.bird_vel

        # Find next tube
        next_tube = next((t for t in self.tubes if t["x"] + self.TUBE_WIDTH > self.BIRD_X), None)
        if not next_tube:
            next_tube = max(self.tubes, key=lambda t: t["x"])

        next_x = max(0, next_tube["x"] - self.BIRD_X)
        gap_center_y = next_tube["height"] + self.TUBE_GAP / 2

        # Vertical distance (bird center vs gap center) normalized
        vertical_distance = (y + self.BIRD_HEIGHT / 2 - gap_center_y) / self.HEIGHT

        return np.array([
            np.clip(vy / 10, -1.5, 1.5),
            next_x / self.WIDTH,
            np.clip(vertical_distance, -1.5, 1.5)
        ], dtype=np.float32)

    def step(self, action):
        """Action: 0 = no jump, 1 = jump"""
        if self.done:
            return self.get_state(), 0.0, True, {"score": self.score}

        # Execute action
        if action == 1:
            self.bird_vel = self.JUMP_STRENGTH

        # Physics update
        self.Bird_y += self.bird_vel
        self.bird_vel += self.GRAVITY

        # Move tubes
        for t in self.tubes:
            t["x"] -= self.TUBE_VELOCITY
            if t["x"] < -self.TUBE_WIDTH:
                t["x"] = max([tube["x"] for tube in self.tubes]) + 350
                t["height"] = randint(150, 300)
                t["passed"] = False

        # Score update
        passed_tube = False
        for t in self.tubes:
            if not t["passed"] and t["x"] + self.TUBE_WIDTH < self.BIRD_X:
                self.score += 1
                t["passed"] = True
                passed_tube = True

        # Collision detection
        collision = False
        for t in self.tubes:
            if (self.BIRD_X + self.BIRD_WIDTH > t["x"] and 
                self.BIRD_X < t["x"] + self.TUBE_WIDTH):
                if (self.Bird_y < t["height"] or 
                    self.Bird_y + self.BIRD_HEIGHT > t["height"] + self.TUBE_GAP):
                    collision = True
                    break

        # Reward calculation
        if (self.Bird_y < 0 or 
            self.Bird_y + self.BIRD_HEIGHT > self.HEIGHT or 
            collision):
            self.done = True
            reward = -5.0
        else:
            # Survival reward
            reward = 0.2
            
            # Bonus for passing a tube
            if passed_tube:
                reward += 20.0
            
            # Proximity bonus (stay near the gap center)
            next_tube = next((t for t in self.tubes if t["x"] + self.TUBE_WIDTH > self.BIRD_X), self.tubes[0])
            gap_center = next_tube["height"] + self.TUBE_GAP / 2
            distance_to_center = abs(self.Bird_y + self.BIRD_HEIGHT / 2 - gap_center)
            proximity_reward = 3.0 * max(0.0, 1.0 - distance_to_center / (self.HEIGHT / 2))
            reward += proximity_reward

        return self.get_state(), reward, self.done, {"score": self.score}

    def render(self):
        if not self.render_mode:
            return
        self.clock.tick(60)
        self.screen.fill(self.GREEN)

        # Draw tubes
        for t in self.tubes:
            pygame.draw.rect(self.screen, self.BLUE, (t["x"], 0, self.TUBE_WIDTH, t["height"]))
            pygame.draw.rect(
                self.screen,
                self.BLUE,
                (t["x"], t["height"] + self.TUBE_GAP, self.TUBE_WIDTH, self.HEIGHT - t["height"] - self.TUBE_GAP),
            )

        # Draw bird
        pygame.draw.rect(self.screen, self.PINK, (self.BIRD_X, self.Bird_y, self.BIRD_WIDTH, self.BIRD_HEIGHT))

        # Display score
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def close(self):
        pygame.quit()
