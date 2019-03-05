import p5
import math
import random
from Block import Block
from Ball import Ball
from Brain import Manual, Neural_Network, Deep_Learning


class Scene:
    def __init__(self, w=900, h=600):
        # prediction
        self.brain = Neural_Network()
        self.batch_size = 50
        self.epochs = 100

        # global const
        self.width = w
        self.height = h
        p5.size(self.width, self.height)

        self.should_serve = True
        self.paused = True
        self.serve_user = True
        self.user_interaction = 0

        self.gathering_NN_data = False
        self.gathering_DL_data = False
        self.trainings = []

        self.draw_prediction = True

        self.speed_factor = 1.4
        self.block_height = self.height / 5.0
        self.block_width = self.width / 50.0
        self.ball_radius = self.height / 40.0
        self.speed = self.block_height * 0.09

        self.user = Block(0, 0)
        self.machine = Block(self.width - self.block_width, 0)
        self.ball = Ball(0, 0)

        self.serve()
        self.printScore()

    def draw(self):
        if self.paused:
            return

        p5.background(0, 0, 0, 90)

        self.update()

        p5.fill(230)
        p5.stroke(230, 1)

        p5.rect((self.user.x, self.user.y),
                self.block_width, self.block_height)
        p5.rect((self.machine.x, self.machine.y),
                self.block_width, self.block_height)
        p5.ellipse((self.ball.x, self.ball.y), 2 *
                   self.ball_radius, 2 * self.ball_radius)

    def update(self):
        self.user.y += self.user_interaction * self.speed

        if self.user.y > self.height - self.block_height:
            self.user.y = self.height - self.block_height
        elif self.user.y < 0:
            self.user.y = 0

        y_pred = self.predictCollisionHeight()
        if self.draw_prediction:
            p5.no_fill()
            p5.stroke(220, 5)
            p5.ellipse((self.width - self.block_width - self.ball_radius, y_pred),
                       2 * self.ball_radius, 2 * self.ball_radius)

        machine_interaction = 0
        if self.machine.y + 0.5 * self.block_height > y_pred + 0.125 * self.block_height:
            machine_interaction -= 1
        elif self.machine.y + 0.5 * self.block_height < y_pred - 0.125 * self.block_height:
            machine_interaction += 1
        self.machine.y += machine_interaction * self.speed

        if self.machine.y > self.height - self.block_height:
            self.machine.y = self.height - self.block_height
        elif self.machine.y < 0:
            self.machine.y = 0

        # speed adjustment
        if math.fabs(self.ball.vy) <= self.speed_factor * self.speed * math.sin(math.pi / 30):
            s = math.copysign(1, self.ball.vy)
            v = self.speed_factor * self.speed * \
                math.log(math.sqrt(1 + math.pow(random.gauss(0, 1), 2)))
            # NOTE altering & random
            self.ball.vy += v * round(random.random()) if s == 0 else s

        if math.fabs(self.ball.vx) <= self.speed_factor * self.speed * math.sin(math.pi / 30):
            s = math.copysign(1, self.ball.vx)
            v = self.speed_factor * self.speed * \
                math.log(math.sqrt(1 + math.pow(random.gauss(0, 1), 2)))
            # NOTE altering & random
            self.ball.vx += v * round(random.random()) if s == 0 else s

        d = math.sqrt(self.ball.vx * self.ball.vx +
                      self.ball.vy * self.ball.vy)
        self.ball.vx *= self.speed_factor * self.speed / d
        self.ball.vy *= self.speed_factor * self.speed / d

        # ball
        self.ball.x += self.ball.vx
        self.ball.y += self.ball.vy

        # user collision
        if self.ball.x <= self.block_width + self.ball_radius and self.ball.y >= self.user.y and self.ball.y <= self.user.y + self.block_height:
            self.ball.x = 2 * (self.block_width +
                               self.ball_radius) - self.ball.x
            self.ball.vx = math.fabs(self.ball.vx)
            self.ball.vy += self.speed_factor * self.speed * self.user_interaction * \
                math.pow(random.random(), 0.25) / 5  # NOTE altering & random
        # machine collision
        if self.ball.x >= self.width - self.block_width - self.ball_radius and self.ball.y >= self.machine.y and self.ball.y <= self.machine.y + self.block_height:
            self.ball.x = 2 * \
                (self.width - self.block_width - self.ball_radius) - self.ball.x
            self.ball.vx = -math.fabs(self.ball.vx)
            self.ball.vy += self.speed_factor * self.speed * machine_interaction * \
                math.pow(random.random(), 0.25) / 5  # NOTE altering & random

        # edge collision
        if self.ball.y < self.ball_radius:
            self.ball.y = 2 * self.ball_radius - self.ball.y
            self.ball.vy = math.fabs(self.ball.vy)
        elif (self.ball.y > self.height - self.ball_radius):
            self.ball.y = 2 * (self.height - self.ball_radius) - self.ball.y
            self.ball.vy = -math.fabs(self.ball.vy)

        if self.ball.x < self.ball_radius:  # machine get a point
            self.machine.score += 1
            self.printScore()
            self.should_serve = True
            self.pause(True)
        elif self.ball.x > self.width - self.ball_radius:  # user get a point
            self.user.score += 1
            self.printScore()
            self.should_serve = True
            self.pause(True)

    def key_pressed(self, event):
        # print(event.key)
        # print(event.is_shift_down())

        if event.key == ' ':
            self.pause(not self.paused)
            if self.should_serve:
                self.serve()

        if event.is_shift_down() and event.key == 'UP':
            if self.batch_size < 700:
                self.batch_size += 10
            print("increasing batch_size to {}".format(self.batch_size))
        elif event.key == 'UP':
            print("decreasing u_i")
            self.user_interaction -= 1

        if event.is_shift_down() and event.key == 'DOWN':
            if self.batch_size > 10:
                self.batch_size -= 10
            print("decreasing batch_size to {}".format(self.batch_size))
        elif event.key == 'DOWN':
            print("increasing u_i")
            self.user_interaction += 1

        if event.is_shift_down() and event.key == 'LEFT':
            if self.epochs > 50:
                self.epochs -= 50
            print("decreasing epochs to {}".format(self.epochs))

        if event.is_shift_down() and event.key == 'RIGHT':
            if self.epochs < 600:
                self.epochs += 50
            print("increasing epochs to {}".format(self.epochs))

        if event.key == 't':
            print('training @{}bs&{}ep'.format(self.batch_size, self.epochs))
            self.brain.train(self.batch_size, self.epochs, verbose_=1)

    def key_released(self, event):
        # print(event.key)
        if event.key == 'UP':
            self.user_interaction += 1
        if event.key == 'DOWN':
            self.user_interaction -= 1

    def pause(self, arg):
        if not arg is None:
            if arg:
                self.paused = True
            else:
                self.paused = False

        return self.paused

    def printScore(self):
        p5.title("{} - {}".format(self.user.score, self.machine.score))

    def resetScore(self):
        self.user.score = 0
        self.machine.score = 0

    def serve(self):
        if not self.should_serve:
            return

        angle = 0.5 * math.pi * (random.random() - 0.5)

        self.ball.x = self.block_width + self.ball_radius
        self.ball.y = self.user.y + 0.5 * self.block_height

        if not self.serve_user:
            self.ball.x = self.width - self.block_width - self.ball_radius
            self.ball.y = self.machine.y + 0.5 * self.block_height
            angle += math.pi

        self.ball.vx = self.speed_factor * self.speed * math.cos(angle)
        self.ball.vy = self.speed_factor * self.speed * math.sin(angle)

        self.serve_user = not self.serve_user
        self.should_serve = False

    def predictCollisionHeight(self):
        W = self.width - 2 * (self.ball_radius + self.block_width)
        H = self.height - 2 * self.ball_radius

        x0 = (self.ball.x - self.ball_radius - self.block_width) / W
        y0 = (self.ball.y - self.ball_radius) / H

        if self.brain.useRaw:
            pass
        else:
            yC = self.brain.predict(
                x0,
                y0,
                self.ball.vx / self.speed / self.speed_factor,
                self.ball.vy / self.speed / self.speed_factor,
                H / W
            )

        return yC * H + self.ball_radius
