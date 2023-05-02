from tkinter import PhotoImage, Tk, Canvas, ttk, BOTH, RIDGE, Label, RAISED, NO, IntVar, Checkbutton
from tkinter.ttk import Frame
import sys, os
from tkinter.filedialog import askopenfilename
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, savgol_filter


class OutputFrame(Frame):
    def __init__(self, window):
        # Сделаем рамку для вывода графиков, рисунков и т. п.
        super().__init__(window)

        # Сделаем заголовок
        self.lable = Label(self)
        self.lable.config(
            text="ИНТЕГРАЛЬНЫЙ КАЛОРИМЕТР",
            relief=RIDGE,
            font=('Artifakt Element', 30),
            borderwidth=4
            )
        self.lable.pack(side='top', expand=True, fill=BOTH)

        # Создадим поле холста
        self.canvas = Canvas(self)
        self.canvas.config(
            bg='#446',
            relief=RAISED,
            width=1000,
            height=840
            )
        self.canvas.pack(side='left', fill=BOTH)

        # Разместим изображение схемы калориметра
        self.img = PhotoImage(file=resource_path('calorimeter_geometry.png'), height=840, width=1100)
        self.canvas.create_image(500, 420, image=self.img, anchor='center')


class AppWindow(Tk):
    # Инициализируем окно
    def __init__(self):
        super().__init__()

        self.title("Юстировка калориметра")
        self.geometry('1600x2560')

        # self.resizable(width=False, height=False)
        self.geometry('1438x1000')


def block_creating():
    global tree

    # Создадим кнопки, заголовки и таблицу блока взаимодействия
    # Добавим заголовок
    _lable = Label(
        window,
        text="УПРАВЛЕНИЕ",
        relief=RIDGE,
        font=('Artifakt Element', 30),
        width=25,
        borderwidth=4
        )
    _lable.pack(side='top')

    # Зададим стиль текста кнопок в панели управления
    _style = ttk.Style()
    _style.theme_use('classic')
    _style.configure(
        'W.TButton',
        foreground='black',
        font=('Artifakt Element', 20)
        )

    # Добавим кнопку демонстрации точки столкновения
    point_show_button = ttk.Button(
        window,
        text="Показать точку столкновения",
        width=25,
        style='W.TButton',
        command=data_analysis
        )
    point_show_button.pack(side='top', ipadx=34, ipady=20)

    # Добавим кнопку смены набора данных
    change_data_pack_button = ttk.Button(
        window,
        text="Выбрать данные",
        width=25,
        style='W.TButton',
        command=data_loading
        )
    change_data_pack_button.pack(side='top', ipadx=34, ipady=20)

    # Добавим кнопку построения графиков
    plot_graphs_button = ttk.Button(
        window,
        text="Построить зависимости",
        width=25,
        style='W.TButton',
        command=data_plotting
        )
    plot_graphs_button.pack(side='top', ipadx=34, ipady=20)

    # Создадим стиль таблицы
    _style.configure(
        "mystyle.Treeview",
        highlightthickness=0,
        bd=0,
        font=('Artifakt Element', 11)
        )
    _style.configure(
        "mystyle.Treeview.Heading",
        font=('Artifakt Element', 13, 'bold')
        )

    # Добавим таблицу с максимальными значениями температур и энергиями по каждому каналу
    columns = ("number", "value", "energy")
    tree = ttk.Treeview(
        columns=columns,
        show="headings", 
        style='mystyle.Treeview'
        )
    tree.pack(fill='y', expand=False, side='left')

    # Определим заголовки
    tree.heading("number", text="№")
    tree.heading("value", text="T_max, град. C")
    tree.heading("energy", text="Q, кДж")

    # Зададим параметры колонок
    tree.column("#1", stretch=NO, width=140)
    tree.column("#2", stretch=NO, width=182)
    tree.column('#3', stretch=NO, width=100)

    window.mainloop()

    return tree

def table_calculations(scale_int):
    # Поиск точки попадания пучка
    max_points = [max(data[:, el]) for el in range(0, data.shape[1])]
    max_point_number = max_points.index(max(max_points)) + 1 # номер термопары

    # Посчитаем энергосодержание потока
    # Найдём точки минимума (список упорядочен по номерам термопар)
    min_points = [min(data[:, el]) for el in range(0, data.shape[1])]

    # Посчитаем производные
    for collumns in range(0, data.shape[1]):
        dU = np.diff(savgol_filter(data[:, collumns], 111, 3))
        derivate = list()

        for el in range(1, data.shape[0]):
            derivate.append(dU[el - 1]/el)

        if collumns == 0:
            summary = np.zeros(len(derivate))
        else:
            summary =+ np.array(derivate)
    
    # Сделаем срез лишней части массива, где производная неопределена
    summary = summary[100::] * 1000

    # Отыщем минимумы производных
    peaks, _ = find_peaks(-1*summary, height=0.02)

    # Теперь получим номер точки, где установилось термодинамическое равновесие
    balance_indx = peaks[-1] + 100 + scale_int # 100 за компенсацию среза
    # Найдём точку термодинамического равновесия по индексу в массиве
    balance_points = [data[balance_indx, el] for el in range(0, data.shape[1])]

    # Посчитаем энергосодержание потока
    energies = list()
    for el in range(0, data.shape[1]):
        delta = balance_points[el] - min_points[el]
        energies.append(round(delta * 390 * 3 * 0.025, 2)) # 0.390 Дж/(кг °С) * 3 кг * 25 °С/мВ

    # Блок анализа зависимостей (для тестирования)
    # plt.plot(summary)
    # plt.plot(peaks, summary[peaks], "x")
    # plt.plot(np.zeros_like(summary), "--", color="gray")
    # plt.show()

    # plt.plot(balance_indx * np.ones_like(balance_points), balance_points, "x")
    # data_plotting()

    # Сделаем список из кортежей
    table_data = list()
    indx = 1
    for point in max_points:
        table_data.append((indx, round(point*25, 2), energies[indx - 1])) # умножили на коэффициент перевода в температуру
        indx += 1

    # Очистим содержиое таблицы
    tree.delete(*tree.get_children())

    # Выведем данные в таблицу
    for el in table_data:
        tree.insert("", 'end', values=el)

    return max_point_number

def thermocouples_location(max_point_number):
    # Построим сетку термопар на схеме
    thermocouples = {
        "1" : [False, 650, 515],
        "2" : [True, 865, 402],
        "3" : [False, 480, 585],
        "4" : [True, 655, 402],
        "5" : [False, 465, 225],
        "6" : [False, 785, 345],
        "7" : [False, 645, 290],
        "8" : [False, 240, 680],
        "9" : [False, 535, 245],
        "10" : [True, 780, 402],
        "11" : [False, 240, 120],
        "12" : [False, 790, 460],
        "13" : [True, 475, 402],
        "14" : [True, 580, 260]
    }

    for number, coordinates in thermocouples.items():
        is_dashed, x, y = coordinates

        # Зададим условие на цвет точки попадания пучка
        if number == str(max_point_number):
            color = "red"
        else:
            color = "blue"

        # Проверка условия на размещение в переднем/заднем плане
        if is_dashed:
            frame.canvas.create_line(x, y, x + 30, y + 30, activefill="green", fill=color, width=5, dash=2)

            frame.canvas.create_line(x + 30, y, x, y + 30, activefill="green", fill=color, width=5, dash=2)
        else:
            frame.canvas.create_line(x, y, x + 30, y + 30, activefill="green", fill=color, width=5)

            frame.canvas.create_line(x + 30, y, x, y + 30, activefill="green", fill=color, width=5)

        # Добавим номера каналов
        if number in ["8", "3", "1", "12", "10"]:
            frame.canvas.create_text(x - 25, y + 30, font=('Artifakt Element', 30), text=number, fill="black")
        elif number == "6":
            frame.canvas.create_text(x - 25, y + 10, font=('Artifakt Element', 30), text=number, fill="black")
        elif number == "2":
            frame.canvas.create_text(x + 55, y - 15, font=('Artifakt Element', 30), text=number, fill="black")
        else:
            frame.canvas.create_text(x + 15, y + 45, font=('Artifakt Element', 30), text=number, fill="black")

def data_analysis(scale_int=560):
    try:
        # Рассчитаем данные для таблицы
        max_point_number = table_calculations(scale_int)

        # Выведем в окно сетку термопар
        thermocouples_location(max_point_number)
    except NameError:
        # Подгрузим данные
        data_loading()

        # Рассчитаем данные для таблицы
        max_point_number = table_calculations(scale_int)

        # Выведем в окно сетку термопар
        thermocouples_location(max_point_number)
    except FileNotFoundError:
        pass

def data_loading():
    global data, check_button_list

    # Запросим директорию файла с данными
    directory = askopenfilename()

    # Импортируем данные калориметра
    data = np.genfromtxt(directory, delimiter=';', skip_header=True, skip_footer=True)
        
    return data

def plots():
    # Построим графики по всем каналам 
    indx = 1
    colors = ['black', 'red', 'blue', 'yellow', 'green', '#8A2BE2', 'pink', 'violet', '#FF4500', '#C71585', '#FF8C00', '#BDB76B', '#4B0082', '#00FF00', '#00FFFF']
    for collumns in range(0, data.shape[1]):
        x = data[:, collumns]
        # peaks, _ = find_peaks(x, height=0.2, width=100, prominence=1)
        # plt.plot(peaks, x[peaks], "x")
        plt.plot(x, label=f"Термопара {indx}", color=colors[indx - 1])
        plt.plot(np.zeros_like(x), "--", color="gray")
        indx += 1

    # Выведем графики в окно
    plt.legend()
    plt.title("Результаты измерений")
    plt.show()

def data_plotting():
    try:
        # Выведем зависимости в графическое окно
        plots()
    except NameError:
        # Подгрузим данные
        data_loading()

        # Выведем зависимости в графическое окно
        plots()
    except FileNotFoundError:
        pass

def resource_path(relative_path):
    """Возвращает обсолютный путь объекта, работает для PyInstaller при компиляции в один файл"""

    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def main():
    global frame, window

    # Инициализация рабочей области
    window = AppWindow()

    # Инициализация окна вывода
    frame = OutputFrame(window)
    frame.config(borderwidth=4, relief=RIDGE)
    frame.pack(side='left', fill=BOTH, expand=True)

    # Инициализация блока управления
    block_creating()

if __name__ == "__main__":
    main()