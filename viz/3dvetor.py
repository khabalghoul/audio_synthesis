from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, NodePath, Vec3
class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.camera.setPos(0, -10, 0)  # Move the camera back a bit
        self.camera.lookAt(0, 0, 0)  # Have the camera look at the origin

        self.draw_axes()
        self.draw_vector(3, 3, 3)

    def draw_axes(self):
        axis = LineSegs()
        axis.setThickness(2)

        # X axis in red
        axis.setColor(1, 0, 0, 1)
        axis.moveTo(0, 0, 0)
        axis.drawTo(10, 0, 0)  # Extended length for visibility

        # Y axis in green
        axis.setColor(0, 1, 0, 1)
        axis.moveTo(0, 0, 0)
        axis.drawTo(0, 10, 0)

        # Z axis in blue
        axis.setColor(0, 0, 1, 1)
        axis.moveTo(0, 0, 0)
        axis.drawTo(0, 0, 10)

        axisNP = NodePath(axis.create())
        axisNP.reparentTo(self.render)

    def draw_vector(self, x, y, z):
        vector = LineSegs()
        vector.setThickness(2)
        vector.setColor(0, 200/255.0, 0, 1)  # Different green shade
        vector.moveTo(0, 0, 0)
        vector.drawTo(x, y, z)

        vectorNP = NodePath(vector.create())
        vectorNP.reparentTo(self.render)

app = MyApp()
app.run()
