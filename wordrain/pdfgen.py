from . import FONT_NORMAL, FONT_EMPHASIZED, FONT_NEWWORD

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab.rl_config
reportlab.rl_config.warnOnMissingFontGlyphs = 0

import matplotlib.transforms
from matplotlib.markers import MarkerStyle

DEBUG = False

fontnames = {}
fontstyle_to_fonttype = {
    ('normal', 'normal'): FONT_NORMAL,
    ('normal', 'bold'): FONT_EMPHASIZED,
    ('italic', 'normal'): FONT_NEWWORD,
}

def registerFonts(fontlist):
    for fonttype in fontlist:
        fontname = fonttype
        font = TTFont(fontname, fontlist[fonttype])
        pdfmetrics.registerFont(font)
        fontnames[fonttype] = fontname

def registerDefaultFonts(fontlist):
    for fonttype in fontlist:
        if fonttype in fontnames:
            continue
        fontname = fonttype
        font = TTFont(fontname, fontlist[fonttype])
        pdfmetrics.registerFont(font)
        fontnames[fonttype] = fontname

class TextShim():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        pass
    def get_window_extent(self, **kwargs):
        xmin = self.x
        ymin = self.y
        xmax = self.x + self.width
        ymax = self.y + self.height
        return matplotlib.transforms.Bbox([[xmin, ymin], [xmax, ymax]])

class AxesShim():
    def __init__(self, pagesize, matplotshim, compact=False):
        self.matplotshim = matplotshim
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.xminmargin = 0.125
        self.xmaxmargin = 0.1
        self.yminmargin = 0.11
        self.ymaxmargin = 0.12
        self.compact = compact
        if self.compact:
            self.ymaxmargin = 0
        self.page_width, self.page_height = pagesize
    def set_xlim(self, **kwargs):
        self.xmin = kwargs["xmin"]
        self.xmax = kwargs["xmax"]
        return
    def set_ylim(self, **kwargs):
        self.ymin = kwargs["ymin"]
        self.ymax = kwargs["ymax"]
        return
    def set_axis_off(self):
        pass
    @property
    def xscale(self):
        logic_width = self.xmax - self.xmin
        usable_page_width = self.page_width * (1 - self.xminmargin - self.xmaxmargin)
        return usable_page_width / logic_width
    @property
    def yscale(self):
        logic_height = self.ymax - self.ymin
        usable_page_height = self.page_height * (1 - self.yminmargin - self.ymaxmargin)
        return usable_page_height / logic_height
    @property
    def centerx(self):
        return (self.xmin + self.xmax) / 2
    @property
    def titlepos(self):
        if self.compact:
            return (self.xmin, self.ymax)
        else:
            return (self.centerx, self.ymax)
    @property
    def xoffset(self):
        return self.page_width * self.xminmargin
    @property
    def yoffset(self):
        return self.page_height * self.yminmargin
    @property
    def transData(self):
        xscale = self.xscale
        yscale = self.yscale
        xmin_scaled = self.xmin * xscale
        ymin_scaled = self.ymin * yscale
        return matplotlib.transforms.Affine2D.from_values(xscale, 0, 0, yscale, -xmin_scaled+self.xoffset, -ymin_scaled+self.yoffset)
    def axvline(self, x, y0, y1, **kwargs):
        return self.matplotshim.axvline(x, y0, y1, **kwargs)
    def plot(self, xcoords, ycoords, **kwargs):
        return self.matplotshim.plot(xcoords, ycoords, **kwargs)
    def scatter(self, x, y, **kwargs):
        return self.matplotshim.scatter(x, y, **kwargs)
    def text(self, x, y, s, **kwargs):
        return self.matplotshim.text(x, y, s, **kwargs)
    def underline(self, x, y, s, **kwargs):
        return self.matplotshim.underline(x, y, s, **kwargs)
    def set_title(self, title, **kwargs):
        return self.matplotshim.set_title(title, **kwargs)


from reportlab.lib.colors import Color

def parse_color(color_arg):
    if isinstance(color_arg, tuple) or isinstance(color_arg, list):
        if len(color_arg) == 3:
            r,g,b = color_arg
            return Color(r,g,b)
        elif len(color_arg) == 4:
            r,g,b,a = color_arg
            return Color(r,g,b,alpha=a)
        else:
            raise ValueError("Unhandled color value: " + repr(color_arg))
    else:
        raise ValueError("Unhandled color value: " + repr(color_arg))

class BoundingBox():
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def update(self, x, y):
        if self.xmin is None:
            self.xmin = x
        else:
            self.xmin = min(self.xmin, x)
        if self.xmax is None:
            self.xmax = x
        else:
            self.xmax = max(self.xmax, x)
        if self.ymin is None:
            self.ymin = y
        else:
            self.ymin = min(self.ymin, y)
        if self.ymax is None:
            self.ymax = y
        else:
            self.ymax = max(self.ymax, y)

    def update_x(self, x):
        if self.xmin is None:
            self.xmin = x
        else:
            self.xmin = min(self.xmin, x)
        if self.xmax is None:
            self.xmax = x
        else:
            self.xmax = max(self.xmax, x)

    def update_rect(self, x, y, width, height):
        self.update(x, y)
        self.update(x+width, y+height)

    def update_bbox(self, bbox):
        self.update(bbox.xmin, bbox.ymin)
        self.update(bbox.xmax, bbox.ymax)

    def update_circle(self, x, y, r):
        self.update(x-r, y-r)
        self.update(x+r, y+r)

    def extend(self, radius):
        self.xmin -= radius
        self.xmax += radius
        self.ymin -= radius
        self.ymax += radius

    def cropbox(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def rectbox(self):
        return (self.xmin, self.ymin, self.xmax-self.xmin, self.ymax-self.ymin)

def lineTo(path, x, y, move=False, bbox=None):
    if move:
        path.moveTo(x, y)
    else:
        path.lineTo(x, y)
    if bbox:
        bbox.update(x, y)

import math

def ngonpath(path, cx, cy, r, n, bbox=None):
    tau = math.pi*2
    angle_down = tau*3/4
    angle_step = tau / n
    for i in range(n):
        angle = angle_step * i + angle_down
        if i == 0:
            lineTo(path, cx+math.cos(angle)*r, cy+math.sin(angle)*r, move=True, bbox=bbox)
        else:
            lineTo(path, cx+math.cos(angle)*r, cy+math.sin(angle)*r, bbox=bbox)
    path.close()

def starpath(path, cx, cy, r, n, bbox=None):
    tau = math.pi*2
    angle_down = tau*3/4
    angle_step = tau / n / 2
    for i in range(n*2):
        angle = angle_step * i + angle_down
        if i == 0:
            lineTo(path, cx+math.cos(angle)*r, cy+math.sin(angle)*r, move=True, bbox=bbox)
        else:
            if i % 2 == 0:
                lineTo(path, cx+math.cos(angle)*r, cy+math.sin(angle)*r, bbox=bbox)
            else:
                lineTo(path, cx+math.cos(angle)*r/2, cy+math.sin(angle)*r/2, bbox=bbox)
    path.close()

class MatplotlibShimPdf():
    def __init__(self, filename, pagesize = (460,345), subgraphs=1, compact=False):
        self.compact = compact
        if compact:
            self.xmargin = 20
            self.ymargin = 20
        else:
            self.xmargin = 0
            self.ymargin = 0
        self.pagesize = pagesize
        self.real_pagesize = (pagesize[0]+self.xmargin*2,pagesize[1]*subgraphs+self.ymargin*2)
        self.subgraphs = subgraphs
        self.yoffset = 0
        self.bbox = BoundingBox()
        self.uncommitted_vlines = []
        self.committed_vlines = []
        if filename is None:
            self.canvas = None
        else:
            self.canvas = canvas.Canvas(filename, pagesize=self.real_pagesize)
        self._axes = AxesShim(self.pagesize, self, compact=compact)
        self.drawcommands = {}
    def _get_renderer(self):
        return None
    def clf(self):
        pass
    def gca(self):
        return self._axes
    def set_axis_off(self):
        pass
    def set_subgraph(self, subgraph):
        if self.bbox.ymin is None or not self.compact:
            self.yoffset = (self.subgraphs - subgraph - 1) * self.pagesize[1]
        else:
            self.yoffset = self.bbox.ymin - self.pagesize[1]
        for vline in self.uncommitted_vlines:
            vline["ylimit"] = self.bbox.ymin
            self.committed_vlines.append(vline)
        self.uncommitted_vlines = []

    def axvline(self, x, y0, y1, **kwargs):
        pdfx, pdfy0 = self.getpdfcoords(x, y0)
        _, pdfy1 = self.getpdfcoords(x, y1)
        color = parse_color(kwargs.get("color"))
        linewidth=kwargs.get("linewidth", 1)
        self.uncommitted_vlines.append({
            "linewidth": linewidth,
            "color": color,
            "pdfx": pdfx,
            "pdfy0": pdfy0,
            "pdfy1": pdfy1,
            "yoffset": self.yoffset,
        })
    def register_point_x(self, x):
        pdfx, pdfy = self.getpdfcoords(x, 0)
        self.bbox.update_x(pdfx)
    def text(self, x, y, s, **kwargs):
        fontsize = kwargs["fontsize"]
        fontweight = kwargs.get("fontweight", "normal")
        fontstyle = kwargs.get("fontstyle", "normal")
        fonttype = fontstyle_to_fonttype[(fontstyle, fontweight)]
        fontname = fontnames.get(fonttype)
        if fontname is None:
            raise Exception("No font file registered for variant "+repr((fontstyle, fontweight)))
        pdfx, pdfy = self.getpdfcoords(x, y)
        width = pdfmetrics.stringWidth(s, fontname, fontsize)
        ascent, descent = pdfmetrics.getAscentDescent(fontname, fontsize)
        height = (ascent - descent)
        zorder = kwargs.get("zorder", 0)
        background = kwargs.get("background")
        horizontalalignment = kwargs.get("horizontalalignment", "left")
        if horizontalalignment == "right":
            stringCommand = "drawRightString"
        else:
            stringCommand = "drawString"
        self.drawcommands.setdefault(zorder, []).append([
            ("setFont", [fontname, fontsize], {}),
            ("setFillColor", [Color(0,0,0)], {}),
            (stringCommand, [pdfx,pdfy-ascent+self.yoffset,s], {}),
        ])
        bbox = BoundingBox()
        bbox.update(pdfx, pdfy-height+self.yoffset)
        if horizontalalignment == "right":
            bbox.update(pdfx-width, pdfy+self.yoffset)
        else:
            bbox.update(pdfx+width, pdfy+self.yoffset)
        if self.canvas is not None and background is not None:
            path = self.canvas.beginPath()
            path.rect(*bbox.rectbox())
            self.drawcommands.setdefault(zorder-1, []).append([
                ("setFillColor", [parse_color(background)], {}),
                ("path", [path], {"stroke":0, "fill":1}),
            ])
        self.bbox.update_bbox(bbox)
        return TextShim(pdfx, pdfy-height, width, height)
    def underline(self, x, y, s, **kwargs):
        fontsize = kwargs["fontsize"]
        fontweight = kwargs.get("fontweight", "normal")
        fontstyle = kwargs.get("fontstyle", "normal")
        fonttype = fontstyle_to_fonttype[(fontstyle, fontweight)]
        fontname = fontnames.get(fonttype)
        if fontname is None:
            raise Exception("No font file registered for variant "+repr((fontstyle, fontweight)))
        pdfx, pdfy = self.getpdfcoords(x, y)
        width = pdfmetrics.stringWidth(s, fontname, fontsize)
        ascent, descent = pdfmetrics.getAscentDescent(fontname, fontsize)
        height = (ascent - descent)
        zorder = kwargs.get("zorder", 0)
        background = kwargs.get("background")
        horizontalalignment = kwargs.get("horizontalalignment", "left")
        path = self.canvas.beginPath()
        y = pdfy-ascent+self.yoffset - 2*fontsize/30
        if horizontalalignment == "right":
            stringCommand = "drawRightString"
            path.rect(pdfx,y,-width,fontsize/30)
            self.bbox.update_rect(pdfx,y,-width,fontsize/30)
        else:
            stringCommand = "drawString"
            path.rect(pdfx,y,width,fontsize/30)
            self.bbox.update_rect(pdfx,y,width,fontsize/30)
        self.drawcommands.setdefault(zorder, []).append([
            ("setFont", [fontname, fontsize], {}),
            ("setFillColor", [Color(0,0,0)], {}),
            ("path", [path], {"stroke":0, "fill":1}),
        ])
    def getpdfcoords(self, x, y):
        transform = self._axes.transData
        pdfx, pdfy = transform.transform((x, y))
        return pdfx, pdfy
    def plot(self, xcoords, ycoords, **kwargs):
        from_x, to_x = xcoords
        from_y, to_y = ycoords
        pdf_from_x, pdf_from_y = self.getpdfcoords(from_x, from_y)
        pdf_to_x, pdf_to_y = self.getpdfcoords(to_x, to_y)
        color = parse_color(kwargs.get("color"))
        zorder = kwargs.get("zorder")
        linewidth=kwargs.get("linewidth", 1)
        self.drawcommands.setdefault(zorder, []).append([
            ("setLineWidth", [linewidth], {}),
            ("setStrokeColor", [color], {}),
            ("line", [pdf_from_x,pdf_from_y+self.yoffset,pdf_to_x,pdf_to_y+self.yoffset], {}),
        ])
        self.bbox.update(pdf_from_x, pdf_from_y+self.yoffset)
        self.bbox.update(pdf_to_x,pdf_to_y+self.yoffset)

    def scatter(self, x, y, **kwargs):
        if self.canvas is None:
            return
        pdf_x, pdf_y = self.getpdfcoords(x, y)
        color = parse_color(kwargs.get("color"))
        marker = kwargs["marker"]
        zorder = kwargs.get("zorder", 0)
        s=kwargs.get("s", 1)
        if marker == "o":
            s = min(s, 0.3)
            linewidth = max(s*4-0.3, 0.1)
            self.drawcommands.setdefault(zorder, []).append([
                ("setLineWidth", [linewidth], {}),
                ("setStrokeColor", [color], {}),
                ("setFillColor", [color], {}),
                ("circle", [pdf_x, pdf_y+self.yoffset, 0.5], {"stroke":1, "fill":0}),
                ("circle", [pdf_x, pdf_y+self.yoffset, s*2], {"stroke":0, "fill":1}),
            ])
            self.bbox.update_circle(pdf_x, pdf_y+self.yoffset, 0.5)
            self.bbox.update_circle(pdf_x, pdf_y+self.yoffset, s*2)
        elif marker == ".":
            s = min(s, 0.3)
            linewidth = max(s*4-0.3, 0.1)
            self.drawcommands.setdefault(zorder, []).append([
                ("setLineWidth", [linewidth], {}),
                ("setStrokeColor", [color], {}),
                ("setFillColor", [color], {}),
                ("circle", [pdf_x, pdf_y+self.yoffset, s*2], {"stroke":0, "fill":1}),
            ])
            self.bbox.update_circle(pdf_x, pdf_y+self.yoffset, s*2)
        elif isinstance(marker, MarkerStyle) and marker.get_marker() == "v":
            s = min(s, 0.6)
            s = max(s, 0.2)
            linewidth = max(s*2, 1)
            path = self.canvas.beginPath()
            bbox = BoundingBox()
            xmax = ngonpath(path, pdf_x, pdf_y+self.yoffset, s, 3, bbox=bbox)
            self.drawcommands.setdefault(zorder, []).append([
                ("setLineWidth", [linewidth], {}),
                ("setStrokeColor", [color], {}),
                ("setFillColor", [color], {}),
                ("setLineJoin", [1], {}),
                ("path", [path], {"stroke":1, "fill":1}),
            ])
            bbox.extend(linewidth)
            self.bbox.update_bbox(bbox)
        elif marker == "*":
            s = min(s, 0.6)
            s = max(s, 0.2)
            linewidth = max(s*2, 1)
            path = self.canvas.beginPath()
            bbox = BoundingBox()
            starpath(path, pdf_x, pdf_y+self.yoffset, s, 5, bbox=bbox)
            self.drawcommands.setdefault(zorder, []).append([
                ("setLineWidth", [linewidth], {}),
                ("setStrokeColor", [color], {}),
                ("setFillColor", [color], {}),
                ("setLineJoin", [1], {}),
                ("path", [path], {"stroke":1, "fill":1}),
            ])
            bbox.extend(linewidth)
            self.bbox.update_bbox(bbox)
        else:
            raise ValueError("Unhandled marker value: " + repr(marker))
        pass
    def set_title(self, title, **kwargs):
        fontsize = kwargs["fontsize"]
        x, y = self._axes.titlepos
        pdfx, pdfy = self.getpdfcoords(x, y)
        fontname = fontnames.get(FONT_NORMAL)
        width = pdfmetrics.stringWidth(title, fontname, fontsize)
        ascent, descent = pdfmetrics.getAscentDescent(fontname, fontsize)
        height = (ascent - descent)
        zorder = kwargs.get("zorder", 0)
        if self.compact:
            rotation = 90
        else:
            rotation = 0
        if self.compact:
            widthadjust = -width
            heightadjust = -descent
        else:
            widthadjust = -width / 2
            heightadjust = -ascent
        pdfy += self.yoffset
        translation = [pdfx,pdfy]
        self.drawcommands.setdefault(zorder, []).append([
            ("saveState", [], {}),
            ("setFont", [fontname, fontsize], {}),
            ("setFillColor", [Color(0,0,0)], {}),
            ("translate", translation, {}),
            ("rotate", [rotation], {}),
            ("drawString", [widthadjust,height*2+heightadjust,title], {}),
            ("restoreState", [], {}),
        ])
        bbox = BoundingBox()
        if self.compact:
            bbox.update(pdfx-height*2, pdfy-width)
            bbox.update(pdfx-height*3, pdfy)
        else:
            bbox.update(pdfx-width/2, pdfy+height)
            bbox.update(pdfx+width/2, pdfy+height*2)
        self.bbox.update_bbox(bbox)

    def savefig(self, *args, **kwargs):
        if DEBUG:
            pdfmaxx, pdfmaxy = self.getpdfcoords(self._axes.xmax, self._axes.ymax)
            pdfminx, pdfminy = self.getpdfcoords(self._axes.xmin, self._axes.ymin)
            self.canvas.setLineWidth(0.1)
            self.canvas.setStrokeColor(Color(0,255,0))
            self.canvas.rect(pdfminx, pdfminy, pdfmaxx-pdfminx, pdfmaxy-pdfminy)

        self.yoffset = self.bbox.ymin - self.pagesize[1]
        for vline in self.uncommitted_vlines:
            vline["ylimit"] = self.bbox.ymin
            self.committed_vlines.append(vline)
        self.uncommitted_vlines = []
        for vline in self.committed_vlines:
            ylimit = vline["ylimit"]
            yoffset = vline["yoffset"]
            linewidth = vline["linewidth"]
            pdfx = vline["pdfx"]
            pdfy0 = vline["pdfy0"]
            pdfy1 = vline["pdfy1"]
            ndots = int((pdfy1 - pdfy0) / (linewidth*3))
            self.canvas.setLineWidth(linewidth)
            self.canvas.setStrokeColor(vline["color"])
            for i in range(ndots):
                y = i * linewidth * 3 + pdfy0 + yoffset
                if y < ylimit and self.compact:
                    continue
                self.canvas.line(pdfx,y,pdfx,y+linewidth)
        for zorder in sorted(self.drawcommands.keys()):
            for commandlist in self.drawcommands[zorder]:
                for drawcommand, drawargs, drawkwargs in commandlist:
                    if drawcommand == "circle":
                        self.canvas.circle(*drawargs, **drawkwargs)
                    elif drawcommand == "setFillColor":
                        self.canvas.setFillColor(*drawargs, **drawkwargs)
                    elif drawcommand == "setLineWidth":
                        self.canvas.setLineWidth(*drawargs, **drawkwargs)
                    elif drawcommand == "setStrokeColor":
                        self.canvas.setStrokeColor(*drawargs, **drawkwargs)
                    elif drawcommand == "setLineJoin":
                        self.canvas.setLineJoin(*drawargs, **drawkwargs)
                    elif drawcommand == "line":
                        self.canvas.line(*drawargs, **drawkwargs)
                    elif drawcommand == "path":
                        self.canvas.drawPath(*drawargs, **drawkwargs)
                    elif drawcommand == "drawString":
                        self.canvas.drawString(*drawargs, **drawkwargs)
                    elif drawcommand == "drawRightString":
                        self.canvas.drawRightString(*drawargs, **drawkwargs)
                    elif drawcommand == "setFont":
                        self.canvas.setFont(*drawargs, **drawkwargs)
                    elif drawcommand == "saveState":
                        self.canvas.saveState(*drawargs, **drawkwargs)
                    elif drawcommand == "restoreState":
                        self.canvas.restoreState(*drawargs, **drawkwargs)
                    elif drawcommand == "rotate":
                        self.canvas.rotate(*drawargs, **drawkwargs)
                    elif drawcommand == "translate":
                        self.canvas.translate(*drawargs, **drawkwargs)
                    else:
                        assert False

        self.drawcommands = {}
        if self.compact:
            self.bbox.extend(20)
            self.canvas.setCropBox(self.bbox.cropbox())
#        print("cropbox", self.bbox.cropbox())
        if DEBUG:
            self.canvas.setLineWidth(0.1)
            self.canvas.setStrokeColor(Color(0,0,255))
            self.canvas.rect(*self.bbox.rectbox())
        self.canvas.showPage()
        self.canvas.save()

