
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

def loadProgram(path, ctx):
    f = open(path, 'r')
    fstr = ''.join(f.readlines())
    return cl.Program(ctx, fstr).build()
def float3buf(f4count, ctx):
    bytesPerFloat = 4
    return [
        cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, f4count * bytesPerFloat),
        cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, f4count * bytesPerFloat),
        cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, f4count * bytesPerFloat)]


def together(*el):
    final = []
    for x in el:
        if type(x) is list or type(x) is tuple:
            final.extend(x)
        else:
            final.append(x)
    return final
def test_cl(maxIterations):
    ctx = cl.create_some_context()#(interactive=False)

    #print 'ctx', ctx
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    buildSceneProg = loadProgram('buildScene.cl', ctx)
    emitInitialRaysProg = loadProgram('emitInitialRays.cl', ctx)
    intersectProg = loadProgram('intersect.cl', ctx)
    genShadowRaysProg = loadProgram('genShadowRay.cl', ctx)
    intersectShadowProg = loadProgram('intersectShadow.cl', ctx)
    bounceProg = loadProgram('bounce.cl', ctx)

    cameraPos = np.array([0,0,.5,1])
    invView = la.inv(look_at((0,0,.5), (0,0,100), (0,1,0)))
    invProj = la.inv(perspective(60, 1, 1, 1000))
    print 'view', invView
    print 'proj', invProj
    viewParamsData = cameraPos.flatten().tolist() + np.transpose(invView).flatten().tolist() + np.transpose(invProj).flatten().tolist()
    #print 'vpd', viewParamsData
    viewParams = struct.pack('4f16f16f', *viewParamsData)
    viewParams_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=viewParams)

    dest = np.zeros((1000, 1000, 4), dtype=np.float16)    
    dest_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, dest.nbytes)

    iterationDest = np.ndarray((1000, 1000, 4), dtype=np.float16)    
    iterationDest_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, iterationDest.nbytes)

    vpshape = (dest.shape[0], dest.shape[1])
    linshape = (vpshape[0] * vpshape[1], 1)

    sceneBuf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 4 * (3*(5) + 1*(2))) # hard coded buffer size

    numPixels = 1000 * 1000
    lightRayPositionBufs = float3buf(numPixels, ctx)
    lightRayDirectionBufs = float3buf(numPixels, ctx)
    shadowRayPositionBufs = float3buf(numPixels, ctx)
    shadowRayDirectionBufs = float3buf(numPixels, ctx)

    hitPositionBufs = float3buf(numPixels, ctx)
    hitNormalBufs = float3buf(numPixels, ctx)
    initialThroughputs = np.ones((1000, 1000, 4), dtype=np.float16)
    throughputBufs = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=initialThroughputs);

    expectedTBuf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, numPixels * 4)
    obstructedBuf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, numPixels * 4)
    materialIdBuf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, numPixels * 4)

    #process
    #emit initial rays
    #   intersect w/ scene
    #   gen shadow rays
    #   intersect shadow rays
    #   bounce

    totalBounces = 3

    buildScene_evt = buildSceneProg.buildScene(queue, (1, 1), None,
        *together(sceneBuf))



    bounceParams = struct.pack('ii', 0, 0)
    bounceParamsBuf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bounceParams)


    for iterationIdx in range(0, maxIterations):
        print 'iteration', iterationIdx

        emitInitialRays_evt = emitInitialRaysProg.emitInitialRays(queue, vpshape, None, 
            *together(viewParams_buf, lightRayPositionBufs, lightRayDirectionBufs, 
                iterationDest_buf, throughputBufs))
        for bounceIdx in range(0, totalBounces):

            bounceParams = struct.pack('ii', iterationIdx, bounceIdx)
            cl.enqueue_write_buffer(queue, bounceParamsBuf, bounceParams)

            intersect_evt = intersectProg.intersect(queue, vpshape, None, 
                *together(sceneBuf, lightRayPositionBufs, lightRayDirectionBufs, hitNormalBufs, hitPositionBufs, materialIdBuf))

            genShadowRay_evt = genShadowRaysProg.genShadowRay(queue, vpshape, None, 
                *together(hitPositionBufs, shadowRayPositionBufs, shadowRayDirectionBufs, expectedTBuf))

            intersectShadow_evt = intersectShadowProg.intersectShadow(queue, vpshape, None, 
                *together(sceneBuf, shadowRayPositionBufs, shadowRayDirectionBufs, expectedTBuf, obstructedBuf))

            if bounceIdx < totalBounces - 1:
                bounceFinal_evt = bounceProg.bounce(queue, vpshape, None,
                    *together(bounceParamsBuf, obstructedBuf, hitNormalBufs, 
                    hitPositionBufs, materialIdBuf, lightRayPositionBufs, lightRayDirectionBufs,  
                    throughputBufs, iterationDest_buf))

        bounceParams = struct.pack('ii', iterationIdx, totalBounces - 1)
        cl.enqueue_write_buffer(queue, bounceParamsBuf, bounceParams)
        bounceFinal_evt = bounceProg.bounceFinal(queue, vpshape, None,
            *together(bounceParamsBuf, obstructedBuf, hitNormalBufs, 
                hitPositionBufs, materialIdBuf, throughputBufs, iterationDest_buf, dest_buf))
        queue.finish()
    

    debug = np.zeros(vpshape, dtype=np.float32)
    cl.enqueue_read_buffer(queue, dest_buf, dest).wait()
    #cl.enqueue_read_buffer(queue, hitNormalBufs[1], debug).wait()q
    '''
    profileEvents = {
        'buildScene': buildScene_evt,
        'emitInitialRays': emitInitialRays_evt,
        'intersect': intersect_evt,
        'genShadowRay': genShadowRay_evt,
        'intersectShadow': intersectShadow_evt,
        'bounceFinal': bounceFinal_evt
    }


    totalTime = 0
    for name, evt in profileEvents.iteritems():
        time = (evt.profile.end - evt.profile.start)* 0.000001
        totalTime += time
        print name, time , 'ms'
    print 'total time', totalTime
    '''

    #return debug
    return dest

if __name__ == "__main__":
    test_cl(10)
    #p2 = window()
