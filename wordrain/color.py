import math
import numpy as np

# https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB
illuminant_D65 = np.array([
    95.0489,
    100,
    108.8840,
])

# https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB
illuminant_D50 = np.array([
    96.4212,
    100,
    82.5188,
])

# https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
xyz_d65_to_linear = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])

# https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIELAB_to_CIEXYZ
lab_to_xyz = np.array([
    [1/116, 1/116, 1/116],
    [1/500, 0, 0],
    [0, 0, -1/200],
])

lab_pre_add = np.array([16,0,0])

oklab_m1 = np.array([
    [0.8189330101,0.3618667424,-0.1288597137],
    [0.0329845436,0.9293118715,0.0361456387],
    [0.0482003018,0.2643662691,0.633851707]
]).transpose()

oklab_m1inv = np.array([
    [1.2270138511,-0.55779998065,0.28125614897],
    [-0.040580178423,1.1122568696,-0.071676678666],
    [-0.076381284506,-0.42148197842,1.5861632204]
]).transpose()
oklab_m2 = np.array([
    [0.2104542553,0.793617785,-0.0040720468],
    [1.9779984951,-2.428592205,0.4505937099],
    [0.0259040371,0.7827717662,-0.808675766]
]).transpose()
oklab_m2inv = np.array([
    [0.99999999845,0.39633779217,0.21580375806],
    [1.0000000089,-0.10556134232,-0.063854174772],
    [1.0000000547,-0.089484182095,-1.2914855379]
]).transpose()

@np.vectorize
def cielab_f_inv(t):
    delta = 6/29
    if t > delta:
        return t**3
    else:
        return 3 * delta**2 * (t - 4/29)

# https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
@np.vectorize
def gamma(c):
    if c > 0.0031308:
        return 1.055 * c ** (1/2.4) - 0.055
    else:
        return 12.92 * c



def xyz_srgb(xyz):
    xyz = np.vstack(xyz)
    rgb = np.matmul(xyz_d65_to_linear, xyz).transpose()[0]
    return gamma(rgb)

def lab_xyz(Lab):
    XYZ_pre = np.matmul(Lab + lab_pre_add, lab_to_xyz)
    XYZ = illuminant_D65 * cielab_f_inv(XYZ_pre)
    return XYZ / 100

def oklab_xyz(Lab):
    lms_prime = np.matmul(Lab, oklab_m2inv)
    lms = np.power(lms_prime, 3)
    XYZ = np.matmul(lms, oklab_m1inv)
    return XYZ

def lchab_lab(L, C, h):
    a = C * math.cos(h/180*math.pi)
    b = C * math.sin(h/180*math.pi)
    return np.array([L, a, b])

def sin_band_filter(v, from_value, to_value):
    if v < from_value or v > to_value:
        return 0
    bandwidth = to_value - from_value
    scaled = (v - from_value) / bandwidth
    result = 0.5-(math.cos(scaled * math.pi * 2) / 2)
    return result

def full_spectrum_map(cmap_value, lightness=0.75, chroma=0.25, startangle=36, adjust_yellow=True):
    hue = cmap_value*360+startangle
    if adjust_yellow:
        adjust_value = sin_band_filter(hue % 360.0, 60, 130)
    else:
        adjust_value = 0
    lab = lchab_lab(lightness + adjust_value * 0.1, chroma , hue)
    xyz = oklab_xyz(lab)
    srgb = xyz_srgb(xyz)
    return list(np.clip(srgb, 0, 1)) + [1.0]

if __name__ == '__main__':
    for i in range(20):
        print(full_spectrum_map(i/20))

    from PIL import Image, ImageFont, ImageDraw, ImageEnhance

    image = Image.new("RGB", (1130, 169))
    imagedraw = ImageDraw.Draw(image)
    imagedraw.rectangle((0, 0) + image.size, fill=(255,255,255))
    for i in range(1130):
        r, g, b, a = full_spectrum_map(i/1130)
        r, g, b = int(r*255), int(g*255), int(b*255)
        imagedraw.rectangle((i, 0) + (i+1,169), fill=(r,g,b))
    outputimage = open("debug.png", "wb")
    image.save(outputimage, "PNG")
