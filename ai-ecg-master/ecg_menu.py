import numpy as np
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib.artist as artist
import matplotlib.image as image

class ItemProperties(object):
    def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow',
                 alpha=1.0):
        self.fontsize = fontsize
        self.labelcolor = labelcolor
        self.bgcolor = bgcolor
        self.alpha = alpha

        self.labelcolor_rgb = colors.to_rgba(labelcolor)[:3]
        self.bgcolor_rgb = colors.to_rgba(bgcolor)[:3]

class MenuItem(artist.Artist):
    padx = 10
    pady = 10

    def __init__(self, fig, labelstr, props=None, hoverprops=None,
                 on_select=None):
        super().__init__()

        self.set_figure(fig)
        self.labelstr = labelstr

        if props is None:
            props = ItemProperties()

        if hoverprops is None:
            hoverprops = ItemProperties()

        self.props = props
        self.hoverprops = hoverprops

        self.on_select = on_select

        # Use fig.text for label rendering (figure coordinates)
        # Initial position will be updated in set_extent
        self.label = fig.text(0, 0, labelstr,
                             fontsize=props.fontsize,
                             color=props.labelcolor,
                             verticalalignment='top',
                             horizontalalignment='left',
                             zorder=10)

        # Estimate label size (approximate)
        self.labelwidth = props.fontsize * len(labelstr) // 2
        self.labelheight = props.fontsize + 4
        self.depth = self.labelheight

        self.rect = patches.Rectangle((0, 0), 1, 1)

        self.set_hover_props(False)

        fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        over, junk = self.rect.contains(event)
        if not over:
            return

        if self.on_select is not None:
            self.on_select(self)

    def set_extent(self, x, y, w, h):
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w)
        self.rect.set_height(h)
        self.rect.set_edgecolor('none')
        self.rect.set_linewidth(0)

        # Center label on box using normalized figure coordinates
        fig = self.figure
        fig_width, fig_height = fig.get_size_inches() * fig.dpi
        center_x = x + w / 2
        center_y = y + h / 2
        norm_x = center_x / fig_width
        norm_y = center_y / fig_height
        self.label.set_position((norm_x, norm_y))
        self.label.set_horizontalalignment('center')
        self.label.set_verticalalignment('center')

        self.hover = False

    def draw(self, renderer):
        self.rect.draw(renderer)
        self.label.draw(renderer)

    def set_hover_props(self, b):
        if b:
            props = self.hoverprops
        else:
            props = self.props

        self.label.set_color(props.labelcolor)
        self.label.set_fontsize(props.fontsize)
        self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

    def set_hover(self, event):
        'check the hover status of event and return true if status is changed'
        b, junk = self.rect.contains(event)

        changed = (b != self.hover)

        if changed:
            self.set_hover_props(b)

        self.hover = b
        return changed


class Menu(object):
    def __init__(self, fig, menuitems, x0, y0):
        self.figure = fig
        self.menuitems = menuitems
        self.numitems = len(menuitems)

        # Create an overlay axes for the menu bar
        self.menu_ax = fig.add_axes([0, 0.94, 1, 0.06], frameon=False)
        self.menu_ax.set_xticks([])
        self.menu_ax.set_yticks([])
        self.menu_ax.set_xlim(0, 1)
        self.menu_ax.set_ylim(0, 1)
        self.menu_ax.set_facecolor('none')

        n = self.numitems
        button_width = 1.0 / n
        button_height = 1.0
        for i, item in enumerate(menuitems):
            left = i * button_width
            # Set extent in axes coordinates
            x_ax = left
            y_ax = 0
            w_ax = button_width
            h_ax = button_height
            # Draw rectangle and label in menu_ax
            rect = patches.Rectangle((x_ax, y_ax), w_ax, h_ax, facecolor=item.props.bgcolor, alpha=item.props.alpha, edgecolor='none', zorder=1)
            self.menu_ax.add_patch(rect)
            item.rect = rect
            label = self.menu_ax.text(x_ax + w_ax/2, y_ax + h_ax/2, item.labelstr,
                                      fontsize=item.props.fontsize,
                                      color=item.props.labelcolor,
                                      ha='center', va='center', zorder=2)
            item.label = label
            item._menu_ax = self.menu_ax

        fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        fig.canvas.mpl_connect('button_release_event', self.on_click)

    def on_move(self, event):
        if event.inaxes != self.menu_ax:
            return
        for item in self.menuitems:
            contains, _ = item.rect.contains(event)
            if contains:
                item.label.set_color(item.hoverprops.labelcolor)
                item.rect.set_facecolor(item.hoverprops.bgcolor)
                item.rect.set_alpha(item.hoverprops.alpha)
            else:
                item.label.set_color(item.props.labelcolor)
                item.rect.set_facecolor(item.props.bgcolor)
                item.rect.set_alpha(item.props.alpha)
        self.figure.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.menu_ax:
            return
        for item in self.menuitems:
            contains, _ = item.rect.contains(event)
            if contains and item.on_select:
                item.on_select(item)
                break
