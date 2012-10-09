
from __future__ import division # 2/3 = .666...
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from vector import *
import numpy as np
import numpy.linalg as la
import pyopencl as cl
import sys
import glutil
import struct
import itertools

def normalize(v):
    return v / la.norm(v)

def look_at(eye, center, up):
    eye = eye if type(eye) is np.array else np.array(eye)
    center = center if type(center) is np.array else np.array(center)
    up = up if type(up) is np.array else np.array(up)

    f = normalize(center - eye)
    u = normalize(up)
    s = normalize(np.cross(f, u))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0,0] = s[0]
    m[1,0] = s[1]
    m[2,0] = s[2]
    m[0,1] = u[0]
    m[1,1] = u[1]
    m[2,1] = u[2]
    m[0,2] =-f[0]
    m[1,2] =-f[1]
    m[2,2] =-f[2]
    m[3,0] =-np.dot(s, eye)
    m[3,1] =-np.dot(u, eye)
    m[3,2] = np.dot(f, eye)
    return m

def perspective(fov_degrees, ar, zn, zf):
    range = math.tan(math.radians(fov_degrees / 2)) * zn    
    left = -range * ar;
    right = range * ar;
    bottom = -range;
    top = range;

    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = (2 * zn) / (right - left)
    m[1,1] = (2 * zn) / (top - bottom)
    m[2,2] = - (zf + zn) / (zf - zn)
    m[2,3] = - 1;
    m[3,2] = - (2 * zf * zn) / (zf - zn);
    return m;

class window(object):
    def __init__(self, *args, **kwargs):
        #mouse handling for transforming scene
        self.mouse_down = False
        self.mouse_old = Vec([0., 0.])
        self.rotate = Vec([0., 0., 0.])
        self.translate = Vec([0., 0., 0.])
        self.initrans = Vec([0., 0., -2.])

        self.width = 640
        self.height = 480

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.win = glutCreateWindow(b"Part 2: Python")

        #gets called by GLUT every frame
        glutDisplayFunc(self.draw)

        #handle user input
        glutKeyboardFunc(self.on_key)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_motion)
        
        #this will call draw every 30 ms
        glutTimerFunc(1, self.timer, 1)

        #setup OpenGL scene
        self.glinit()


        glutMainLoop()       

    def glinit(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., self.width / float(self.height), .1, 1000.)
        glMatrixMode(GL_MODELVIEW)


    ###GL CALLBACKS
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            sys.exit()
        elif args[0] == 't':
            print self.cle.timings

    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old.x = x
        self.mouse_old.y = y

    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old.x
        dy = y - self.mouse_old.y
        if self.mouse_down and self.button == 0: #left button
            self.rotate.x += dy * .2
            self.rotate.y += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate.z -= dy * .01 
        self.mouse_old.x = x
        self.mouse_old.y = y
    ###END GL CALLBACKS


    def draw(self):
        """Render the particles"""        
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        glTranslatef(self.initrans.x, self.initrans.y, self.initrans.z)
        glRotatef(self.rotate.x, 1, 0, 0)
        glRotatef(self.rotate.y, 0, 1, 0) #we switched around the axis so make this rotate_z
        glTranslatef(self.translate.x, self.translate.y, self.translate.z)
        

        #draw the x, y and z axis as lines
        glutil.draw_axes()

        glutSwapBuffers()


def test_cl():
    ctx = cl.create_some_context()#(interactive=False)

    #print 'ctx', ctx
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    f = open('part1.cl', 'r')
    fstr = ''.join(f.readlines())
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags

    cameraPos = np.array([0,3,-3,0])
    invView = la.inv(look_at((0,3, -3), (0,0,0), (0,1,0)))
    invProj = la.inv(perspective(60, 1, 1, 1000))
    print 'view', invView
    print 'proj', invProj
    viewParamsData = cameraPos.flatten().tolist() + np.transpose(invView).flatten().tolist() + np.transpose(invProj).flatten().tolist()
    #print 'vpd', viewParamsData
    viewParams = struct.pack('4f16f16f', *viewParamsData)
    viewParams_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=viewParams)
    num_pixels = 1000 * 1000
    #setup opencl
    dest = np.ndarray((1000, 1000, 4), dtype=np.float32)    
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, dest.nbytes)
    local_shape = (8, 8)
    #run kernel
    evt = program.part1(queue, (dest.shape[0], dest.shape[1]), None, 
        viewParams_buf, dest_buf)
    #evt = program.part1(queue, dest.shape, None, dest_buf)
    cl.enqueue_read_buffer(queue, dest_buf, dest).wait()
    print 'time', (evt.profile.end - evt.profile.start) * 0.000001, 'ms'
    return dest

if __name__ == "__main__":
    test_cl()
    #p2 = window()
