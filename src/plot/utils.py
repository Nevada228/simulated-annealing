import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from typing import List


class PlotData:
    def __init__(self, xdata, ydata, color="red", dotted=False, label=""):
        self.xdata = xdata
        self.ydata = ydata
        self.label = label
        self.dotted = dotted
        self.color = color


class PlotUtils:
    def __init__(self):
        pass

    @staticmethod
    def build_plot(datas: List[PlotData], checkbox=True):
        fig, ax = plt.subplots()
        lines = []
        for data in datas:
            line, = ax.plot(data.xdata, data.ydata, 'xb-', lw=2, color=data.color, label=data.label)
            lines.append(line)
        plt.subplots_adjust(left=0.2)
        if checkbox:
            def change_state(label):
                index = labels.index(label)
                lines[index].set_visible(not lines[index].get_visible())
                plt.draw()

            rax = plt.axes([0.05, 0.4, 0.1, 0.15])
            labels = [str(line.get_label()) for line in lines]
            visibility = [line.get_visible() for line in lines]
            check = CheckButtons(rax, labels, visibility)
            check.on_clicked(change_state)
            plt.show()
