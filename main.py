from Scene import Scene
import p5


def setup():
    global scene
    scene = Scene(900, 600)


def draw():
    global scene
    scene.draw()


def key_pressed(event):
    global scene
    scene.key_pressed(event)


def key_released(event):
    global scene
    scene.key_released(event)


if __name__ == "__main__":
    p5.run()
    pass
