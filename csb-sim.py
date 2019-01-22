#!/home/inoryy/anaconda3/bin/python

import sys, time
from math import sqrt, acos, pi, cos, sin
from decimal import Decimal, ROUND_HALF_UP
from csb-RL import Agent

class Point(object):
    __slots__ = 'x', 'y'

    def __init__(self, x, y):
        self.x, self.y = x, y

    def dist(self, p):
        return sqrt(self.dist2(p))

    def dist2(self, p):
        return (self.x-p.x)**2 + (self.y - p.y)**2

    def closest(self, a, b):
        da = float(b.y - a.y)
        db = float(a.x - b.x)
        c1 = da*a.x + db*a.y
        c2 = -db*self.x + da*self.y
        det = da**2 + db**2

        if det != 0:
            cx = (da*c1 - db*c2) / det
            cy = (da*c2 + db*c1) / det
        else:
            cx, cy = self.x, self.y

        return Point(cx, cy)

class Unit(Point):
    r = 0.

    __slots__ = 'id', 'vx', 'vy'

    def __init__(self, id, x, y, vx, vy):
        super().__init__(x, y)
        self.id, self.vx, self.vy = id, vx, vy

    def collision(self, u):
        if self.vx == u.vx and self.vy == u.vy:
            return

        dx = self.x - u.x
        dy = self.y - u.y
        myp = Point(dx, dy)
        dvx = self.vx - u.vx
        dvy = self.vy - u.vy
        up = Point(0,0)

        p = up.closest(myp, Point(dx + dvx, dy + dvy))

        pdst = up.dist2(p)
        mypdst = myp.dist2(p)

        if type(u) == Checkpoint:
            sr2 = u.r**2
        else:
            sr2 = (self.r + u.r)**2

        if pdst >= sr2:
            return

        length = sqrt(dvx**2 + dvy**2)

        bdst = sqrt(sr2 - pdst)
        p.x -= bdst * dvx / length
        p.y -= bdst * dvy / length

        if myp.dist2(p) > mypdst:
            return

        pdst = p.dist(myp)

        if pdst > length:
            return

        t = pdst / length

        return Collision(self, u, t)

class Checkpoint(Unit):
    r = 600.

class Pod(Unit):
    r = 400.

    __slots__ = 'angle', 'next_cp', 'checked', 'timeout', \
                'shield', 'has_boost', 'cp_ct'

    def __init__(self, id, x, y, vx, vy, angle, next_cp, boosted):
        super().__init__(id, x, y, vx, vy)
        self.angle, self.next_cp = angle, next_cp
        self.checked = 0
        self.timeout = 100
        self.shield = 0
        self.has_boost = True
        self.cp_ct = 0
        self.boosted = False

    def getInfo(self):
        return [self.x, self.y, self.vx, self.vy, self.angle, self.next_cp, self.shield, self.boosted]
    
    def update(self, data):
        self.x, self.y, self.vx, self.vy, self.angle, self.next_cp = data

    def apply(self, action):
        self.rotate(action.p)

        thrust = action.thrust
        if thrust == 650:
            self.has_boost = False
            self.boosted = True

        if thrust == -1:
            thrust = 0
            self.shield = 4

        self.boost(thrust)

    def rotate(self, p):
        a = self.diff_angle(p)
        a = max(-18., min(18., a))

        self.angle += a
        if self.angle >= 360.:
            self.angle = self.angle - 360.
        elif self.angle < 0.0:
            self.angle += 360.

    def boost(self, thrust):
        if self.shield: return

        ra = self.angle * pi / 180.

        self.vx += cos(ra) * thrust
        self.vy += sin(ra) * thrust

    def move(self, t):
        self.x += self.vx * t
        self.y += self.vy * t

    def end(self):
        self.x = int(Decimal(self.x).quantize(0, rounding = ROUND_HALF_UP))
        self.y = int(Decimal(self.y).quantize(0, rounding = ROUND_HALF_UP))
        self.vx = int(self.vx * 0.85)
        self.vy = int(self.vy * 0.85)
        self.timeout -= 1
        self.shield = max(0, self.shield - 1)

        laps = 3 # todo: always?
        if self.checked >= self.cp_ct*laps:
            self.next_cp = 0
            self.checked = self.cp_ct*laps

    def get_angle(self, p):
        d = self.dist(p)
        dx = float(p.x - self.x) / d
        dy = float(p.y - self.y) / d

        a = acos(dx) * 180.0 / pi
        if dy < 0:
            a = 360. - a

        return a

    def diff_angle(self, p):
        a = self.get_angle(p)
        right = a - self.angle if self.angle <= a else 360. - self.angle + a
        left = self.angle - a if self.angle >= a else self.angle + 360. - a

        if right < left:
            return right
        return -left

    def bounce(self, u):
        if type(u) == Checkpoint:
            self.checked += 1
            self.timeout = 100
            self.next_cp = (self.next_cp + 1) % self.cp_ct
            return

        m1, m2 = 1. + 9.*(self.shield == 4), 1. + 9.*(u.shield == 4)
        mcoeff = (m1 + m2) / (m1 * m2)

        nx = self.x - u.x
        ny = self.y - u.y

        dst2 = nx**2 + ny**2

        dvx = self.vx - u.vx
        dvy = self.vy - u.vy

        prod = nx*dvx + ny*dvy
        fx = (nx*prod)/(dst2*mcoeff)
        fy = (ny*prod)/(dst2*mcoeff)

        self.vx -= fx / m1
        self.vy -= fy / m1
        u.vx += fx / m2
        u.vy += fy / m2

        impulse = sqrt(fx**2 + fy**2)
        if impulse < 120.0:
            fx *= 120.0/impulse
            fy *= 120.0/impulse

        self.vx -= fx / m1
        self.vy -= fy / m1
        u.vx += fx / m2
        u.vy += fy / m2

    def __repr__(self):
        return "{} {} {} {} {} {} {} {}".format(
            self.x, self.y, self.vx, self.vy, self.angle,
            self.next_cp, self.shield, 1 - self.has_boost
        )

class Collision(object):
    __slots__ = 'a', 'b', 't'

    def __init__(self, a, b, t):
        self.a, self.b, self.t = a, b, t

class Action():
    def __init__(self, p, thrust):
        if thrust == "SHIELD":
            thrust = -1
        elif thrust == "BOOST":
            thrust = 650
        else:
            thrust = int(thrust)

        self.p, self.thrust = p, thrust

class Game():
    def __init__(self, laps, checkpoints):
        self.laps = laps
        self.checkpoints = checkpoints

    def play(self, pods):
        def _check_col(t, col, first_col):
            return col and col.t + t < 1.0 and (not first_col or col.t < first_col.t)

        t = 0.0
        while t < 1.0:
            first_col = None
            for i in range(len(pods)):
                for j in range(i+1, len(pods)):
                    col = pods[i].collision(pods[j])
                    if _check_col(t, col, first_col):
                        first_col = col

                col = pods[i].collision(self.checkpoints[pods[i].next_cp])
                if _check_col(t, col, first_col):
                    first_col = col

            if not first_col:
                for pod in pods:
                    pod.move(1.0 - t)
                t = 1.0
            else:
                for pod in pods:
                    pod.move(first_col.t)
                first_col.a.bounce(first_col.b)
                t += first_col.t

        for pod in pods:
            pod.end()

if __name__ == '__main__':
    cp_ct = int(input())
    checkpoints = []
    for i in range(cp_ct):
        x,y = map(int, input().split())
        checkpoints.append(Checkpoint(i,x,y,0,0))
    game = Game(-1, checkpoints)
    turns = int(input())
    pods = []
    times = 0.0
    for q in range(turns):
        for i in range(4):
            if q != 0:
                x, y, vx, vy, angle, ncp, shield, boosted = 0, 0, 0, 0, 0, 0, 0, 0
                angle = float(angle)
            else:
                p = pods[i]
                x, y, vx, vy, angle, ncp, shield, boosted = p.getInfo()
                angle = float(angle)
                
            x, y, vx, vy, ncp, shield, boosted = map(int, [x, y, vx, vy, ncp, shield, boosted])
            if len(pods) < 4:
                pod = Pod(i, x, y, vx, vy, angle, ncp)
                pods.append(pod)
            else:
                pod = pods[i]
                pod.update([x, y, vx, vy, angle, ncp])
            pod.shield = shield
            pod.has_boost = 1 - boosted
            pod.cp_ct = cp_ct

        tic = time.time()

        for i in range(4):
            tx, ty, thrust = input().split()
            p = Point(int(tx), int(ty))
            pods[i].apply(Action(p, thrust))

        game.play(pods)

        toc = time.time()
        times += toc - tic

        for pod in pods:
            print(pod)
    print((times / turns)*1000, file = sys.stderr)
