#!/usr/bin/env python3
# ShanShui.py - A SVG Chinese landscape generator
# Inspired by Lingdong Huang's Shan-shui-inf (https://github.com/LingDong-/shan-shui-inf)

import argparse
import datetime
import math
import base64
import json
import random
import os
import sys
from typing import List, Dict, Tuple, Union, Any, Optional, Callable

# PRNG - Custom random number generator for consistent results
class PRNG:
    def __init__(self, seed=None):
        self.s = 1234
        self.p = 999979  # Prime numbers as specified in original
        self.q = 999983
        self.m = self.p * self.q
        self.seed(seed)
    
    def hash(self, x):
        """Hash function matching JS implementation"""
        y = base64.b64encode(json.dumps(x).encode()).decode()
        z = 0
        for i in range(len(y)):
            z += ord(y[i]) * (128 ** i)
        return z
    
    def seed(self, x=None):
        """Seed the PRNG, matching JS implementation"""
        if x is None:
            x = int(datetime.datetime.now().timestamp() * 1000)
        
        y = 0
        z = 0
        
        def redo():
            nonlocal y, z
            y = (self.hash(x) + z) % self.m
            z += 1
        
        while y % self.p == 0 or y % self.q == 0 or y == 0 or y == 1:
            redo()
        
        self.s = y
        print(f"int seed: {self.s}")
        for i in range(10):
            self.next()
    
    def next(self):
        """Get next random number, matching JS implementation"""
        self.s = (self.s * self.s) % self.m
        return self.s / self.m
    
    def test(self, f=None):
        """Test function to verify distribution"""
        if f is None:
            f = self.next
        chart = [0] * 10
        for i in range(1000000):  # Reduced from 10M for performance
            chart[math.floor(f() * 10)] += 1
        return chart

# Initialize global PRNG instance
prng = PRNG()

# Override Python's random with our PRNG
random.random = lambda: prng.next()

# Perlin Noise implementation
class Noise:
    def __init__(self):
        self.PERLIN_YWRAPB = 4
        self.PERLIN_YWRAP = 1 << self.PERLIN_YWRAPB
        self.PERLIN_ZWRAPB = 8
        self.PERLIN_ZWRAP = 1 << self.PERLIN_ZWRAPB
        self.PERLIN_SIZE = 4095
        self.perlin_octaves = 4
        self.perlin_amp_falloff = 0.5
        self.perlin = None
        
    def scaled_cosine(self, i):
        """Helper function for Perlin noise calculation"""
        return 0.5 * (1.0 - math.cos(i * math.pi))
    
    def noise(self, x, y=0, z=0):
        """3D Perlin noise function matching JS implementation"""
        # Initialize perlin array if needed
        if self.perlin is None:
            self.perlin = [random.random() for _ in range(self.PERLIN_SIZE + 1)]
        
        # Ensure positive coordinates
        if x < 0: x = -x
        if y < 0: y = -y
        if z < 0: z = -z
        
        # Integer and fractional parts
        xi = math.floor(x)
        yi = math.floor(y)
        zi = math.floor(z)
        xf = x - xi
        yf = y - yi
        zf = z - zi
        
        # Initialize variables
        r = 0
        ampl = 0.5
        
        # For each octave
        for o in range(self.perlin_octaves):
            of = xi + (yi << self.PERLIN_YWRAPB) + (zi << self.PERLIN_ZWRAPB)
            
            # Calculate interpolation factors
            rxf = self.scaled_cosine(xf)
            ryf = self.scaled_cosine(yf)
            
            # Sample and interpolate perlin noise values
            n1 = self.perlin[of & self.PERLIN_SIZE]
            n1 += rxf * (self.perlin[(of + 1) & self.PERLIN_SIZE] - n1)
            
            n2 = self.perlin[(of + self.PERLIN_YWRAP) & self.PERLIN_SIZE]
            n2 += rxf * (self.perlin[(of + self.PERLIN_YWRAP + 1) & self.PERLIN_SIZE] - n2)
            
            n1 += ryf * (n2 - n1)
            
            # Handle z dimension
            of += self.PERLIN_ZWRAP
            n2 = self.perlin[of & self.PERLIN_SIZE]
            n2 += rxf * (self.perlin[(of + 1) & self.PERLIN_SIZE] - n2)
            
            n3 = self.perlin[(of + self.PERLIN_YWRAP) & self.PERLIN_SIZE]
            n3 += rxf * (self.perlin[(of + self.PERLIN_YWRAP + 1) & self.PERLIN_SIZE] - n3)
            
            n2 += ryf * (n3 - n2)
            n1 += self.scaled_cosine(zf) * (n2 - n1)
            
            # Accumulate with amplitude
            r += n1 * ampl
            ampl *= self.perlin_amp_falloff
            
            # Bit shifting for each octave
            xi <<= 1
            xf *= 2
            yi <<= 1
            yf *= 2
            zi <<= 1
            zf *= 2
            
            # Handle overflow
            if xf >= 1.0:
                xi += 1
                xf -= 1
            if yf >= 1.0:
                yi += 1
                yf -= 1
            if zf >= 1.0:
                zi += 1
                zf -= 1
        
        return r
    
    def noise_detail(self, lod, falloff):
        """Set noise detail parameters"""
        if lod > 0:
            self.perlin_octaves = lod
        if falloff > 0:
            self.perlin_amp_falloff = falloff
    
    def noise_seed(self, seed):
        """Seed the noise function with LCG (Linear Congruential Generator)"""
        # Implement LCG from original JS
        m = 4294967296  # 2^32
        a = 1664525
        c = 1013904223
        
        # Create LCG state
        z = seed = (seed if seed is not None else random.random() * m) & 0xFFFFFFFF
        
        # Generate new perlin array using LCG
        self.perlin = []
        for i in range(self.PERLIN_SIZE + 1):
            z = (a * z + c) % m
            self.perlin.append(z / m)

# Global noise instance
noise = Noise()

# Utility functions
def distance(p0, p1):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def mapval(value, istart, istop, ostart, ostop):
    """Map a value from one range to another"""
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))

def loop_noise(nslist):
    """Make a noise list loop cleanly"""
    dif = nslist[-1] - nslist[0]
    bds = [100, -100]
    
    for i in range(len(nslist)):
        nslist[i] += (dif * (len(nslist) - 1 - i)) / (len(nslist) - 1)
        if nslist[i] < bds[0]:
            bds[0] = nslist[i]
        if nslist[i] > bds[1]:
            bds[1] = nslist[i]
    
    for i in range(len(nslist)):
        nslist[i] = mapval(nslist[i], bds[0], bds[1], 0, 1)
    
    return nslist

def rand_choice(arr):
    """Choose a random element from an array"""
    return arr[math.floor(len(arr) * random.random())]

def norm_rand(m, M):
    """Generate a random number in range [m,M]"""
    return mapval(random.random(), 0, 1, m, M)

def wtrand(func):
    """Weighted random generator using given probability function"""
    x = random.random()
    y = random.random()
    if y < func(x):
        return x
    else:
        return wtrand(func)

def rand_gaussian():
    """Generate random number with Gaussian distribution"""
    return wtrand(lambda x: math.pow(math.e, -24 * (x - 0.5)**2)) * 2 - 1

def bezier_point(p0, p1, p2, t):
    """Calculate single point on quadratic bezier curve"""
    u = 1 - t
    return [
        u*u*p0[0] + 2*u*t*p1[0] + t*t*p2[0],
        u*u*p0[1] + 2*u*t*p1[1] + t*t*p2[1]
    ]

def bezmh(P, w=1):
    """Generate points along multiple connected bezier curves"""
    # Handle the case where only two points are given
    if len(P) == 2:
        P = [P[0], mid_point(P[0], P[1]), P[1]]
    
    plist = []
    for j in range(len(P) - 2):
        # Get control points
        if j == 0:
            p0 = P[j]
        else:
            p0 = mid_point(P[j], P[j + 1])
        
        p1 = P[j + 1]
        
        if j == len(P) - 3:
            p2 = P[j + 2]
        else:
            p2 = mid_point(P[j + 1], P[j + 2])
        
        # Generate points along this curve segment
        pl = 20  # Number of points per segment
        for i in range(pl + (1 if j == len(P) - 3 else 0)):
            t = i / pl
            u = (1-t)**2 + 2*t*(1-t)*w + t**2
            
            # Calculate point
            plist.append([
                ((1-t)**2*p0[0] + 2*t*(1-t)*p1[0]*w + t**2*p2[0])/u,
                ((1-t)**2*p0[1] + 2*t*(1-t)*p1[1]*w + t**2*p2[1])/u
            ])
    
    return plist

def mid_point(*args):
    """Calculate the midpoint of a set of points"""
    if len(args) == 1 and isinstance(args[0], list):
        plist = args[0]
    else:
        plist = args
    
    if len(plist) == 0:
        return [0, 0]
    
    # Check if plist is a list of points or just a single point
    if isinstance(plist[0], list):
        return [sum(p[0] for p in plist) / len(plist), 
                sum(p[1] for p in plist) / len(plist)]
    elif len(plist) == 2 and not isinstance(plist[0], list):
        # If we're given just two coordinates (not points)
        return [plist[0], plist[1]]
    else:
        # If plist is a single point
        return plist

# PolyTools class (for triangulation and polygon operations)
class PolyTools:
    @staticmethod
    def midPt(*args):
        """Alias for mid_point"""
        return mid_point(*args)
    
    @staticmethod
    def triangulate(plist, args=None):
        """Triangulate a polygon into triangles"""
        if args is None:
            args = {}
        
        area = args.get('area', 100)
        convex = args.get('convex', False)
        optimize = args.get('optimize', True)
        
        def line_expr(pt0, pt1):
            """Line equation parameters from two points"""
            den = pt1[0] - pt0[0]
            m = float('inf') if den == 0 else (pt1[1] - pt0[1]) / den
            k = pt0[1] - m * pt0[0] if m != float('inf') else 0
            return [m, k]
        
        def intersect(ln0, ln1):
            """Check if two line segments intersect"""
            le0 = line_expr(ln0[0], ln0[1])
            le1 = line_expr(ln1[0], ln1[1])
            
            den = le0[0] - le1[0]
            if den == 0:
                return False
            
            x = (le1[1] - le0[1]) / den
            y = le0[0] * x + le0[1]
            
            def on_seg(p, ln):
                """Check if point is on segment (non-inclusive)"""
                return (min(ln[0][0], ln[1][0]) <= p[0] <= max(ln[0][0], ln[1][0]) and
                        min(ln[0][1], ln[1][1]) <= p[1] <= max(ln[0][1], ln[1][1]))
            
            if on_seg([x, y], ln0) and on_seg([x, y], ln1):
                return [x, y]
            return False
        
        def pt_in_poly(pt, poly):
            """Check if point is inside polygon"""
            scount = 0
            for i in range(len(poly)):
                np = poly[i+1 if i < len(poly)-1 else 0]
                sect = intersect([poly[i], np], [pt, [pt[0] + 999, pt[1] + 999]])
                if sect:
                    scount += 1
            return scount % 2 == 1
        
        def ln_in_poly(ln, poly):
            """Check if line is inside polygon"""
            # Create line with slight inset for testing
            lnc = [[0, 0], [0, 0]]
            ep = 0.01
            
            lnc[0][0] = ln[0][0] * (1 - ep) + ln[1][0] * ep
            lnc[0][1] = ln[0][1] * (1 - ep) + ln[1][1] * ep
            lnc[1][0] = ln[0][0] * ep + ln[1][0] * (1 - ep)
            lnc[1][1] = ln[0][1] * ep + ln[1][1] * (1 - ep)
            
            # Check if line intersects any edge
            for i in range(len(poly)):
                np = poly[i+1 if i < len(poly)-1 else 0]
                if intersect(lnc, [poly[i], np]):
                    return False
            
            # Check if midpoint is inside
            mid = PolyTools.midPt(ln)
            if not pt_in_poly(mid, poly):
                return False
            
            return True
        
        def sides_of(polygon):
            """Calculate side lengths of polygon"""
            slist = []
            for i in range(len(polygon)):
                pt = polygon[i]
                np = polygon[i+1 if i < len(polygon)-1 else 0]
                s = math.sqrt((np[0] - pt[0])**2 + (np[1] - pt[1])**2)
                slist.append(s)
            return slist
        
        def area_of(polygon):
            """Calculate area of triangle"""
            if len(polygon) != 3:
                return 0
                
            slist = sides_of(polygon)
            a, b, c = slist
            s = (a + b + c) / 2
            try:
                return math.sqrt(s * (s - a) * (s - b) * (s - c))
            except ValueError:
                return 0
        
        def sliver_ratio(polygon):
            """Quality measure for triangles"""
            A = area_of(polygon)
            P = sum(sides_of(polygon))
            if P == 0:
                return 0
            return A / P
        
        def best_ear(polygon):
            """Find best ear to cut from polygon"""
            cuts = []
            for i in range(len(polygon)):
                pt = polygon[i]
                lp = polygon[i-1 if i > 0 else len(polygon)-1]
                np = polygon[i+1 if i < len(polygon)-1 else 0]
                
                qlist = polygon.copy()
                qlist.pop(i)
                
                if convex or ln_in_poly([lp, np], polygon):
                    ear = [[lp, pt, np], qlist]
                    if not optimize:
                        return ear
                    cuts.append(ear)
            
            if not cuts:
                return [polygon, []]
                
            best = [polygon, []]
            best_ratio = 0
            
            for cut in cuts:
                r = sliver_ratio(cut[0])
                if r >= best_ratio:
                    best = cut
                    best_ratio = r
            
            return best
        
        def shatter(polygon, a):
            """Recursively shatter polygon into smaller triangles"""
            if not polygon:
                return []
                
            if len(polygon) < 3:
                return []
                
            if area_of(polygon) < a:
                return [polygon]
            else:
                slist = sides_of(polygon)
                # Find index of longest side
                ind = max(range(len(slist)), key=lambda i: slist[i])
                nind = (ind + 1) % len(polygon)
                lind = (ind + 2) % len(polygon)
                
                try:
                    mid = PolyTools.midPt([polygon[ind], polygon[nind]])
                except Exception as e:
                    print(f"Error in shatter: {e}, polygon: {polygon}")
                    return []
                
                return (shatter([polygon[ind], mid, polygon[lind]], a) + 
                        shatter([polygon[lind], polygon[nind], mid], a))
        
        # Main triangulation logic
        if len(plist) <= 3:
            return shatter(plist, area)
        else:
            cut = best_ear(plist)
            return shatter(cut[0], area) + PolyTools.triangulate(cut[1], args)

# SVG Creation functions
def poly(plist, args=None):
    """Create SVG polyline element"""
    if args is None:
        args = {}
    
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    fil = args.get('fil', "rgba(0,0,0,0)")
    str_color = args.get('str', fil)
    wid = args.get('wid', 0)
    
    points = " ".join([f"{(p[0] + xof):.1f},{(p[1] + yof):.1f}" for p in plist])
    
    return f'<polyline points="{points}" style="fill:{fil};stroke:{str_color};stroke-width:{wid}"/>'

def stroke(ptlist, args=None):
    """Create a stroke effect from a path"""
    if args is None:
        args = {}
    
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    wid = args.get('wid', 2)
    col = args.get('col', "rgba(200,200,200,0.9)")
    noi = args.get('noi', 0.5)
    out = args.get('out', 1)
    fun = args.get('fun', lambda x: math.sin(x * math.pi))
    
    if len(ptlist) == 0:
        return ""
    
    vtxlist0 = []
    vtxlist1 = []
    vtxlist = []
    n0 = random.random() * 10
    
    for i in range(1, len(ptlist) - 1):
        w = wid * fun(i / len(ptlist))
        w = w * (1 - noi) + w * noi * noise.noise(i * 0.5, n0)
        
        a1 = math.atan2(ptlist[i][1] - ptlist[i - 1][1], ptlist[i][0] - ptlist[i - 1][0])
        a2 = math.atan2(ptlist[i][1] - ptlist[i + 1][1], ptlist[i][0] - ptlist[i + 1][0])
        a = (a1 + a2) / 2
        
        if a < a2:
            a += math.pi
        
        vtxlist0.append([
            ptlist[i][0] + w * math.cos(a),
            ptlist[i][1] + w * math.sin(a)
        ])
        
        vtxlist1.append([
            ptlist[i][0] - w * math.cos(a),
            ptlist[i][1] - w * math.sin(a)
        ])
    
    # Construct the final vertex list
    vtxlist = [ptlist[0]] + vtxlist0 + list(reversed(vtxlist1)) + [ptlist[-1], ptlist[0]]
    
    return poly([p for p in vtxlist], {'xof': xof, 'yof': yof, 'fil': col, 'str': col, 'wid': out})

def blob(x, y, args=None):
    """Create a blob shape"""
    if args is None:
        args = {}
    
    len_val = args.get('len', 20)
    wid = args.get('wid', 5)
    ang = args.get('ang', 0)
    col = args.get('col', "rgba(200,200,200,0.9)")
    noi = args.get('noi', 0.5)
    ret = args.get('ret', 0)
    
    def default_fun(x):
        if x <= 1:
            return math.pow(math.sin(x * math.pi), 0.5)
        else:
            return -math.pow(math.sin((x + 1) * math.pi), 0.5)
    
    fun = args.get('fun', default_fun)
    
    reso = 20.0
    lalist = []
    
    for i in range(int(reso) + 1):
        p = (i / reso) * 2
        xo = len_val / 2 - abs(p - 1) * len_val
        yo = (fun(p) * wid) / 2
        a = math.atan2(yo, xo)
        l = math.sqrt(xo * xo + yo * yo)
        lalist.append([l, a])
    
    nslist = []
    n0 = random.random() * 10
    
    for i in range(int(reso) + 1):
        nslist.append(noise.noise(i * 0.05, n0))
    
    loop_noise(nslist)
    
    plist = []
    for i in range(len(lalist)):
        ns = nslist[i] * noi + (1 - noi)
        nx = x + math.cos(lalist[i][1] + ang) * lalist[i][0] * ns
        ny = y + math.sin(lalist[i][1] + ang) * lalist[i][0] * ns
        plist.append([nx, ny])
    
    if ret == 0:
        return poly(plist, {'fil': col, 'str': col, 'wid': 0})
    else:
        return plist

def div(plist, reso):
    """Divide a path into more points"""
    if len(plist) < 2:
        return plist
    
    tl = (len(plist) - 1) * reso
    lx, ly = 0, 0
    rlist = []
    
    for i in range(0, tl):
        # Get points to interpolate between
        lastp = plist[math.floor(i / reso)]
        nextp = plist[math.ceil(min(i / reso, len(plist) - 1))]
        p = (i % reso) / reso
        
        nx = lastp[0] * (1 - p) + nextp[0] * p
        ny = lastp[1] * (1 - p) + nextp[1] * p
        
        rlist.append([nx, ny])
        lx, ly = nx, ny
    
    if len(plist) > 0:
        rlist.append(plist[-1])
    
    return rlist

def texture(ptlist, args=None):
    """Create texture for landscape elements"""
    if args is None:
        args = {}
    
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    tex = args.get('tex', 400)
    wid = args.get('wid', 1.5)
    len_val = args.get('len', 0.2)
    sha = args.get('sha', 0)
    ret = args.get('ret', 0)
    
    def default_noi(x):
        return 30 / x
    
    noi = args.get('noi', default_noi)
    
    def default_col(x):
        return f"rgba(100,100,100,{(random.random() * 0.3):.3f})"
    
    col = args.get('col', default_col)
    
    def default_dis():
        if random.random() > 0.5:
            return (1 / 3) * random.random()
        else:
            return (1 * 2) / 3 + (1 / 3) * random.random()
    
    dis = args.get('dis', default_dis)
    
    # Make sure we can access each point in ptlist safely
    if len(ptlist) == 0 or len(ptlist[0]) == 0:
        return "" if not ret else []
    
    reso = [len(ptlist), len(ptlist[0])]
    texlist = []
    
    for i in range(tex):
        mid = int(dis() * reso[1])
        hlen = math.floor(random.random() * (reso[1] * len_val))
        
        start = mid - hlen
        end = mid + hlen
        start = min(max(start, 0), reso[1])
        end = min(max(end, 0), reso[1])
        
        layer = (i / tex) * (reso[0] - 1)
        
        texlist.append([])
        for j in range(int(start), int(end)):
            p = layer - math.floor(layer)
            
            # Make sure indices are in range
            floor_layer = min(math.floor(layer), len(ptlist) - 1)
            ceil_layer = min(math.ceil(layer), len(ptlist) - 1)
            j_idx = min(j, len(ptlist[floor_layer]) - 1)
            
            if j_idx >= len(ptlist[ceil_layer]):
                continue
                
            x = ptlist[floor_layer][j_idx][0] * p + ptlist[ceil_layer][j_idx][0] * (1 - p)
            y = ptlist[floor_layer][j_idx][1] * p + ptlist[ceil_layer][j_idx][1] * (1 - p)
            
            noi_val = noi(layer + 1) if callable(noi) else noi
            ns = [
                noi_val * (noise.noise(x, j * 0.5) - 0.5),
                noi_val * (noise.noise(y, j * 0.5) - 0.5)
            ]
            
            texlist[-1].append([x + ns[0], y + ns[1]])
    
    canv = ""
    # SHADE
    if sha:
        for j in range(0, len(texlist), 1 + (1 if sha != 0 else 0)):
            shade_color = "rgba(100,100,100,0.1)"
            if texlist[j]:  # Check if the list is not empty
                canv += stroke(texlist[j], {'xof': xof, 'yof': yof, 'col': shade_color, 'wid': sha})
    
    # TEXTURE
    for j in range(0 + (1 if sha else 0), len(texlist), 1 + (1 if sha else 0)):
        if texlist[j]:  # Check if the list is not empty
            if callable(col):
                color = col(j / len(texlist))
            else:
                color = col
            canv += stroke(texlist[j], {'xof': xof, 'yof': yof, 'col': color, 'wid': wid})
    
    return texlist if ret else canv

# Tree generator class
class Tree:
    @staticmethod
    def tree01(x, y, args=None):
        """Simple tree - style 1"""
        if args is None:
            args = {}
        
        hei = args.get('hei', 50)
        wid = args.get('wid', 3)
        col = args.get('col', "rgba(100,100,100,0.5)")
        noi = args.get('noi', 0.5)
        
        reso = 10
        nslist = []
        for i in range(reso):
            nslist.append([noise.noise(i * 0.5), noise.noise(i * 0.5, 0.5)])
        
        # Extract leaf color from col
        if "rgba(" in col:
            leafcol = col.replace("rgba(", "").replace(")", "").split(",")
        else:
            leafcol = ["100", "100", "100", "0.5"]
        
        canv = ""
        line1 = []
        line2 = []
        
        for i in range(reso):
            nx = x
            ny = y - (i * hei) / reso
            
            # Add leaves
            if i >= reso / 4:
                for j in range(int((reso - i) / 5)):
                    leaf_alpha = (random.random() * 0.2 + float(leafcol[3]))
                    leaf_color = f"rgba({leafcol[0]},{leafcol[1]},{leafcol[2]},{leaf_alpha:.1f})"
                    
                    canv += blob(
                        nx + (random.random() - 0.5) * wid * 1.2 * (reso - i),
                        ny + (random.random() - 0.5) * wid,
                        {
                            'len': random.random() * 20 * (reso - i) * 0.2 + 10,
                            'wid': random.random() * 6 + 3,
                            'ang': (random.random() - 0.5) * math.pi / 6,
                            'col': leaf_color
                        }
                    )
            
            # Add trunk lines
            line1.append([nx + (nslist[i][0] - 0.5) * wid - wid / 2, ny])
            line2.append([nx + (nslist[i][1] - 0.5) * wid + wid / 2, ny])
        
        # Draw trunk
        canv += poly(line1, {'fil': "none", 'str': col, 'wid': 1.5})
        canv += poly(line2, {'fil': "none", 'str': col, 'wid': 1.5})
        
        return canv
    
    @staticmethod
    def tree02(x, y, args=None):
        """Bushy tree - style 2"""
        if args is None:
            args = {}
        
        hei = args.get('hei', 16)
        wid = args.get('wid', 8)
        clu = args.get('clu', 5)
        col = args.get('col', "rgba(100,100,100,0.5)")
        
        # Extract leaf color from col
        if "rgba(" in col:
            leafcol = col.replace("rgba(", "").replace(")", "").split(",")
        else:
            leafcol = ["100", "100", "100", "0.5"]
        
        canv = ""
        for i in range(clu):
            canv += blob(
                x + rand_gaussian() * clu * 4,
                y + rand_gaussian() * clu * 4,
                {
                    'ang': math.pi / 2,
                    'col': col,
                    'fun': lambda x: x <= 1 
                        and math.pow(abs(math.sin(x * math.pi) * x), 0.5) * (1 if math.sin(x * math.pi) * x >= 0 else -1)
                        or -math.pow(abs(math.sin((x - 2) * math.pi * (x - 2))), 0.5) * (1 if math.sin((x - 2) * math.pi * (x - 2)) >= 0 else -1),
                    'wid': random.random() * wid * 0.75 + wid * 0.5,
                    'len': random.random() * hei * 0.75 + hei * 0.5,
                }
            )
        
        return canv

# Mountain generation class
class Mount:
    @staticmethod
    def mountain(xoff, yoff, seed, args=None):
        """Generate a mountain"""
        if args is None:
            args = {}
        
        hei = args.get('hei', 100 + random.random() * 400)
        wid = args.get('wid', 400 + random.random() * 200)
        tex = args.get('tex', 200)
        veg = args.get('veg', True)
        ret = args.get('ret', 0)
        col = args.get('col', None)
        
        seed = seed if seed is not None else 0
        
        canv = ""
        
        # Generate mountain points
        ptlist = []
        h = hei
        w = wid
        reso = [10, 50]
        
        hoff = 0
        for j in range(reso[0]):
            hoff += (random.random() * yoff) / 100
            ptlist.append([])
            for i in range(reso[1]):
                x = (i / reso[1] - 0.5) * math.pi
                y = math.cos(x)
                y *= noise.noise(x + 10, j * 0.15, seed)
                p = 1 - j / reso[0]
                ptlist[j].append([(x / math.pi) * w * p, -y * h * p + hoff])
        
        # WHITE BG - draw mountain silhouette
        bg_path = ptlist[0] + [[0, reso[0] * 4]]
        canv += poly(bg_path, {'xof': xoff, 'yof': yoff, 'fil': "white", 'str': "none"})
        
        # OUTLINE - draw mountain outline
        canv += stroke([pt for pt in ptlist[0]], {
            'xof': xoff, 'yof': yoff, 
            'col': "rgba(100,100,100,0.3)", 
            'noi': 1, 
            'wid': 3
        })
        
        # Add texture
        texture_args = {
            'xof': xoff,
            'yof': yoff,
            'tex': tex,
            'sha': 0,  # Shadow
        }
        if col:
            texture_args['col'] = col
            
        canv += texture(ptlist, texture_args)
        
        return canv
    
    @staticmethod
    def distMount(xoff, yoff, seed, args=None):
        """Generate a distant mountain range"""
        if args is None:
            args = {}
        
        hei = args.get('hei', 300)
        len_val = args.get('len', 2000)
        seg = args.get('seg', 5)
        
        seed = seed if seed is not None else 0
        canv = ""
        span = 10
        
        ptlist = []
        
        # Generate points for distant mountains
        for i in range(int(len_val / span / seg)):
            ptlist.append([])
            for j in range(seg + 1):
                def tran(k):
                    return [
                        xoff + k * span,
                        yoff - hei * noise.noise(k * 0.05, seed) * 
                        math.pow(math.sin((math.pi * k) / (len_val / span)), 0.5)
                    ]
                
                ptlist[-1].append(tran(i * seg + j))
            
            for j in range(int(seg / 2) + 1):
                def tran(k):
                    return [
                        xoff + k * span,
                        yoff + 24 * noise.noise(k * 0.05, 2, seed) * 
                        math.pow(math.sin((math.pi * k) / (len_val / span)), 1)
                    ]
                
                ptlist[-1].insert(0, tran(i * seg + j * 2))
        
        # Generate polygons for each mountain segment
        for mountain in ptlist:
            def get_col(x, y):
                c = int(noise.noise(x * 0.02, y * 0.02, yoff) * 55 + 200)
                return f"rgb({c},{c},{c})"
            
            # Draw silhouette
            canv += poly(mountain, {
                'fil': get_col(*mountain[-1]),
                'str': "none",
                'wid': 1
            })
            
            # Triangulate and draw with texture
            triangles = PolyTools.triangulate(mountain, {
                'area': 100,
                'convex': True,
                'optimize': False
            })
            
            for triangle in triangles:
                m = PolyTools.midPt(triangle)
                co = get_col(m[0], m[1])
                canv += poly(triangle, {'fil': co, 'str': co, 'wid': 1})
        
        return canv

# Main function to generate SVG landscape
def generate_landscape(width=3000, height=800, seed=None):
    """Generate a Chinese landscape painting as SVG"""
    # Initialize with seed
    if seed is not None:
        prng.seed(seed)
    
    # Set up SVG canvas
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">\n')
    
    # Add background - pale beige/rice paper color
    svg += f'<rect width="{width}" height="{height}" fill="rgb(245,245,235)" />\n'
    
    # Generate distant mountains (background)
    svg += Mount.distMount(width//2, height//3, random.random() * 100, {
        'hei': height//8,
        'len': width * 2
    })
    
    # Generate middle-ground mountains
    for i in range(3):
        x_pos = width * (0.2 + 0.6 * random.random())
        y_pos = height * 0.5 
        mountain_height = height * (0.2 + 0.3 * random.random())
        
        svg += Mount.mountain(x_pos, y_pos, i * random.random(), {
            'hei': mountain_height,
            'wid': width//3 + random.random() * width//4
        })
    
    # Generate foreground elements
    for i in range(10):
        if random.random() < 0.7:  # 70% chance for trees
            x_pos = width * random.random()
            y_pos = height * (0.6 + 0.3 * random.random())
            tree_type = random.random()
            
            if tree_type < 0.5:
                svg += Tree.tree01(x_pos, y_pos, {
                    'hei': height * (0.05 + 0.1 * random.random()),
                    'wid': 3 + random.random() * 5
                })
            else:
                svg += Tree.tree02(x_pos, y_pos, {
                    'hei': height * (0.02 + 0.04 * random.random()),
                    'wid': 8 + random.random() * 10,
                    'clu': 3 + int(random.random() * 5)
                })
    
    # Close SVG
    svg += '</svg>'
    return svg

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Chinese landscape painting as SVG")
    parser.add_argument("--width", type=int, default=3000, help="Width of the image (default: 3000)")
    parser.add_argument("--height", type=int, default=800, help="Height of the image (default: 800)")
    parser.add_argument("--seed", type=str, help="Random seed for reproducible generation")
    parser.add_argument("--output", type=str, help="Output file path (default: shanshui_TIMESTAMP.svg)")
    
    args = parser.parse_args()
    
    # Generate landscape
    svg = generate_landscape(args.width, args.height, args.seed)
    
    # Create output filename with timestamp if not specified
    if not args.output:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"shanshui_{timestamp}.svg"
    else:
        output_file = args.output
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(svg)
    
    print(f"Generated landscape saved to {output_file}")
    print(f"Dimensions: {args.width}x{args.height}")
    if args.seed:
        print(f"Seed: {args.seed}")
